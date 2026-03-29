"""
Anti-Gravity Financial Agentic System
Agent Nodes — Each node is a function that reads/writes AgentState.
Uses Google Gemini for LLM calls and RAG for context retrieval.

Design:
  • Brain runs ONCE — scores, sorts agents by severity, allocates budget.
  • Dispatcher chains agents directly: agent₁ → agent₂ → … → final_planner.
  • Each agent receives its allocated budget and must stay within that limit.
"""

import os
import time
import re
import math
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from scoring_engine import FinancialEngine
from rag_setup import retrieve_context
from visualization import create_spider_chart, create_comparison_chart

load_dotenv()


# ── LLM ────────────────────────────────────────────────────────────
def get_llm(temperature: float = 0.3):
    return ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=temperature,
    )

def invoke_with_retry(llm, messages, agent_name="Agent", max_retries=4):
    """Wrapper to handle Gemini free tier rate limits (429) natively by waiting."""
    print(f"\n[🚀 Calling {getattr(llm, 'model', 'LLM')}] Agent: {agent_name}...")
    for attempt in range(max_retries):
        try:
            return llm.invoke(messages)
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                if attempt == max_retries - 1:
                    raise
                wait_time = 60 # Default safe wait
                match = re.search(r"retry in (\d+(?:\.\d+)?)s", err_msg)
                if match:
                    wait_time = math.ceil(float(match.group(1))) + 2
                print(f"⚠️ API Rate limit hit. Waiting {wait_time}s before retrying (Attempt {attempt+1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                raise


# ── Persona ────────────────────────────────────────────────────────
COACH_PERSONA = """You are a **Senior Financial Coach** in the Anti-Gravity Financial System.

Your communication style:
- Professional yet supportive — like a trusted friend who happens to be a CFP
- Direct and actionable — no vague advice, give specific dollar amounts and steps
- Use markdown formatting with headers, bullet points, and bold for emphasis
- Encourage without sugar-coating — be honest about where they stand
- Always explain the WHY behind each recommendation

CRITICAL BUDGET RULE:
You MUST stay within the allocated budget the Brain has assigned to you.
NEVER suggest expenditures that, when combined with other agents' allocations,
would exceed the user's monthly surplus. If your budget is ₹0, focus on
free/no-cost actions only (e.g. "apply for insurance", "rebalance existing funds").
"""


# ── Mapping helpers ────────────────────────────────────────────────
# Maps agent key → the score dimension(s) it owns
AGENT_SCORE_MAP = {
    "savings":    "emergency",
    "debt":       "debt",
    "insurance":  "insurance",
    "investment": "investment",  # also checks retirement
}

# Default priority when scores are tied
DEFAULT_PRIORITY = ["savings", "debt", "insurance", "investment"]


# ══════════════════════════════════════════════════════════════════
# NODE: Input Processor
# ══════════════════════════════════════════════════════════════════
def input_processor(state: dict) -> dict:
    """
    Validate user data. If any required fields are missing,
    set data_complete=False and list the missing fields.
    """
    user_data = state.get("user_data", {})
    required_fields = [
        "monthly_income", "essential_expenses", "liquid_savings",
        "age", "insurance_status",
    ]

    missing = [f for f in required_fields if f not in user_data or user_data[f] is None]

    if missing:
        msg = AIMessage(content=(
            "⚠️ **Missing Information**\n\n"
            "I need a few more details before I can analyze your finances:\n\n"
            + "\n".join(f"- **{f.replace('_', ' ').title()}**" for f in missing)
            + "\n\nPlease fill out the sidebar form completely."
        ))
        return {
            "data_complete": False,
            "messages": [msg],
        }

    # Set defaults for optional fields
    defaults = {
        "total_debt": 0,
        "monthly_debt_payment": 0,
        "debt_interest_rates": [],
        "debt_names": [],
        "debt_balances": [],
        "retirement_savings": 0,
        "annual_retirement_contribution": 0,
        "tax_advantaged_contributions": 0,
        "employer_match_pct": 0,
        "current_equity_pct": 50,
        "annual_income": user_data.get("monthly_income", 0) * 12,
    }
    for key, default in defaults.items():
        if key not in user_data or user_data[key] is None:
            user_data[key] = default

    return {
        "user_data": user_data,
        "data_complete": True,
    }


# ══════════════════════════════════════════════════════════════════
# NODE: Brain (Orchestrator) — runs ONCE
# ══════════════════════════════════════════════════════════════════
def brain_orchestrator(state: dict) -> dict:
    """
    The Brain runs exactly once. It:
      1. Calculates all 6 financial health scores.
      2. Sorts agents by SEVERITY (lowest score first).
         Ties are broken by DEFAULT_PRIORITY order.
      3. Computes the user's monthly surplus (income − expenses).
      4. Allocates that surplus across agents proportionally
         to their severity weight, so total never exceeds surplus.
      5. Writes the execution order and budgets into state.
    """
    if not state.get("data_complete", False):
        return {"agent_execution_order": [], "current_agent_index": 0}

    user_data = state.get("user_data", {})
    engine = FinancialEngine(user_data)
    scores = engine.calculate_all_scores()

    # ── Generate spider chart ──────────────────────────────────
    chart_path = create_spider_chart(scores)

    # ── Monthly surplus (never negative) ───────────────────────
    income = user_data.get("monthly_income", 0)
    expenses = user_data.get("essential_expenses", 0)
    surplus = max(0, income - expenses)

    # ── Severity weights ───────────────────────────────────────
    # For 'investment' agent, use the worse of investment/retirement
    def agent_score(agent_key):
        if agent_key == "investment":
            return min(scores.get("investment", 10), scores.get("retirement", 10))
        score_key = AGENT_SCORE_MAP.get(agent_key, agent_key)
        return scores.get(score_key, 10)

    # Weight = how far from perfect (10). Higher weight = more severe.
    severity_weights = {}
    for agent_key in DEFAULT_PRIORITY:
        score = agent_score(agent_key)
        severity_weights[agent_key] = max(0, 10 - score)

    # ── Sort by severity (desc), ties broken by default priority ─
    agent_order = sorted(
        DEFAULT_PRIORITY,
        key=lambda a: (-severity_weights[a], DEFAULT_PRIORITY.index(a)),
    )

    # ── Budget allocation ──────────────────────────────────────
    total_weight = sum(severity_weights.values())
    budget_allocations = {}

    if total_weight > 0 and surplus > 0:
        for agent_key in agent_order:
            fraction = severity_weights[agent_key] / total_weight
            budget_allocations[agent_key] = round(surplus * fraction, 2)
    else:
        # All scores are perfect or no surplus — give everyone ₹0
        for agent_key in agent_order:
            budget_allocations[agent_key] = 0.0

    # ── Build severity summary ─────────────────────────────────
    def severity_label(v):
        if v < 4:
            return "🔴 Critical"
        if v < 7:
            return "🟡 Needs Work"
        return "🟢 Strong"

    order_display = " → ".join(
        f"**{a.replace('_', ' ').title()}** ({agent_score(a):.1f})"
        for a in agent_order
    )

    budget_table = "\n".join(
        f"| {a.replace('_', ' ').title()} | {agent_score(a):.1f}/10 | "
        f"{severity_label(agent_score(a))} | ₹{budget_allocations[a]:,.0f}/mo |"
        for a in agent_order
    )

    status_msg = AIMessage(content=(
        f"🧠 **Brain Analysis Complete**\n\n"
        f"**Monthly Surplus:** ₹{surplus:,.0f} "
        f"(Income ₹{income:,.0f} − Expenses ₹{expenses:,.0f})\n\n"
        f"| Dimension | Score | Status | Budget |\n"
        f"|-----------|-------|--------|--------|\n"
        f"{budget_table}\n\n"
        f"**Execution Order (most severe first):**\n{order_display}\n\n"
        f"*Agents are chained directly — no return to Brain.*"
    ), name="internal")

    return {
        "scores": scores,
        "agent_execution_order": agent_order,
        "budget_allocations": budget_allocations,
        "available_monthly_surplus": surplus,
        "current_agent_index": 0,
        "spider_chart_path": chart_path,
        "messages": [status_msg],
        "recommendations": {},
    }


# ══════════════════════════════════════════════════════════════════
# NODE: Agent Dispatcher (lightweight router — no LLM call)
# ══════════════════════════════════════════════════════════════════
def agent_dispatcher(state: dict) -> dict:
    """
    Read agent_execution_order + current_agent_index from state.
    Set active_agent to the next agent in the list, or "done".
    This node is fast — no LLM, no heavy computation.
    """
    order = state.get("agent_execution_order", [])
    idx = state.get("current_agent_index", 0)

    if idx < len(order):
        return {"active_agent": order[idx]}
    return {"active_agent": "done"}


# ══════════════════════════════════════════════════════════════════
# Helper: Budget preamble injected into every agent prompt
# ══════════════════════════════════════════════════════════════════
def _budget_context(state: dict, agent_key: str) -> str:
    """Return a budget summary string for the agent's prompt."""
    budget = state.get("budget_allocations", {}).get(agent_key, 0)
    surplus = state.get("available_monthly_surplus", 0)
    allocations = state.get("budget_allocations", {})
    order = state.get("agent_execution_order", [])

    other_budgets = "\n".join(
        f"  - {k.replace('_', ' ').title()}: ₹{v:,.0f}"
        for k, v in allocations.items() if k != agent_key
    )

    return (
        f"## 💰 Your Allocated Budget\n"
        f"- **Your budget:** ₹{budget:,.0f}/month\n"
        f"- **Total user surplus:** ₹{surplus:,.0f}/month\n"
        f"- Other agents' budgets:\n{other_budgets}\n\n"
        f"⚠️ You MUST keep all recommendations within your ₹{budget:,.0f}/month "
        f"budget. The total across ALL agents equals the user's surplus — "
        f"DO NOT exceed it."
    )


def _advance_index(state: dict) -> dict:
    """Increment current_agent_index so the dispatcher picks the next agent."""
    return {"current_agent_index": state.get("current_agent_index", 0) + 1}


# ══════════════════════════════════════════════════════════════════
# NODE: Savings Agent (Emergency Fund)
# ══════════════════════════════════════════════════════════════════
def savings_agent(state: dict) -> dict:
    """Calculate the dollar gap to a 6-month emergency fund and advise."""
    user_data = state.get("user_data", {})
    scores = state.get("scores", {})
    engine = FinancialEngine(user_data)
    gap = engine.get_emergency_gap()

    context = retrieve_context(
        "orchestrator",
        "emergency fund savings strategy financial order of operations"
    )

    llm = get_llm()
    prompt = f"""{COACH_PERSONA}

You are the **Savings Architect** — your mission is to help this user build their emergency fund.

{_budget_context(state, "savings")}

## User's Situation
- Monthly Income: ₹{user_data.get('monthly_income', 0):,.0f}
- Essential Expenses: ₹{user_data.get('essential_expenses', 0):,.0f}/month
- Current Liquid Savings: ₹{user_data.get('liquid_savings', 0):,.0f}
- Emergency Score: {scores.get('emergency', 0):.1f}/10
- **Gap to 6-Month Fund: ₹{gap:,.0f}**

## Knowledge Base Context
{context}

## Your Task
1. Acknowledge their current savings level
2. Explain why a 6-month emergency fund matters (briefly)
3. Calculate a specific monthly savings target WITHIN your allocated budget
4. Provide 3 actionable steps to close the ₹{gap:,.0f} gap
5. Give a timeline estimate based on your budget allocation

Keep response under 400 words. Use markdown formatting.
"""

    response = invoke_with_retry(llm, [HumanMessage(content=prompt)], agent_name="Savings Architect")
    advice = response.content

    recommendations = state.get("recommendations", {})
    recommendations["savings"] = advice

    return {
        "recommendations": recommendations,
        "messages": [AIMessage(content=f"🏦 **Savings Architect Report**\n\n{advice}", name="internal")],
        **_advance_index(state),
    }


# ══════════════════════════════════════════════════════════════════
# NODE: Debt Agent (Debt Shredder)
# ══════════════════════════════════════════════════════════════════
def debt_agent(state: dict) -> dict:
    """Recommend Avalanche vs Snowball based on user's specific debts."""
    user_data = state.get("user_data", {})
    scores = state.get("scores", {})

    context = retrieve_context(
        "debt_shredder",
        "debt elimination avalanche snowball method high interest credit card"
    )

    # Build debt list
    debt_names = user_data.get("debt_names", [])
    debt_balances = user_data.get("debt_balances", [])
    debt_rates = user_data.get("debt_interest_rates", [])
    debt_list = ""
    for i in range(len(debt_names)):
        name = debt_names[i] if i < len(debt_names) else f"Debt {i+1}"
        balance = debt_balances[i] if i < len(debt_balances) else 0
        rate = debt_rates[i] if i < len(debt_rates) else 0
        debt_list += f"- {name}: ₹{balance:,.0f} @ {rate}% APR\n"

    if not debt_list:
        debt_list = (
            f"- Total Debt: ₹{user_data.get('total_debt', 0):,.0f}\n"
            f"- Monthly Payment: ₹{user_data.get('monthly_debt_payment', 0):,.0f}\n"
        )

    llm = get_llm()
    prompt = f"""{COACH_PERSONA}

You are the **Debt Shredder** — your mission is to help this user eliminate their debt.

{_budget_context(state, "debt")}

## User's Debt Profile
{debt_list}
- Monthly Income: ₹{user_data.get('monthly_income', 0):,.0f}
- Current Monthly Debt Payments: ₹{user_data.get('monthly_debt_payment', 0):,.0f}
- Debt Score: {scores.get('debt', 0):.1f}/10

## Knowledge Base Context
{context}

## Your Task
1. Analyze their specific debts — identify the most dangerous ones
2. Recommend either Avalanche or Snowball method (explain why for THEM)
3. Suggest a payment plan WITHIN your allocated budget of ₹{state.get('budget_allocations', {}).get('debt', 0):,.0f}/month
4. Provide a debt-free timeline estimate at that payment level
5. If any rates > 7%, flag them as URGENT

Keep response under 400 words. Use markdown formatting.
"""

    response = invoke_with_retry(llm, [HumanMessage(content=prompt)], agent_name="Debt Shredder")
    advice = response.content

    recommendations = state.get("recommendations", {})
    recommendations["debt"] = advice

    return {
        "recommendations": recommendations,
        "messages": [AIMessage(content=f"🔥 **Debt Shredder Report**\n\n{advice}", name="internal")],
        **_advance_index(state),
    }


# ══════════════════════════════════════════════════════════════════
# NODE: Insurance Agent
# ══════════════════════════════════════════════════════════════════
def insurance_agent(state: dict) -> dict:
    """Flag missing insurance and provide actionable guidance."""
    user_data = state.get("user_data", {})
    scores = state.get("scores", {})
    insurance_status = user_data.get("insurance_status", {})

    context = retrieve_context(
        "insurance_expert",
        "health life disability insurance coverage requirements"
    )

    llm = get_llm()
    prompt = f"""{COACH_PERSONA}

You are the **Insurance Expert** — your mission is to ensure this user has proper financial protection.

{_budget_context(state, "insurance")}

## User's Insurance Status
- Health Insurance: {'✅ Covered' if insurance_status.get('health') else '❌ NOT COVERED'}
- Life Insurance: {'✅ Covered' if insurance_status.get('life') else '❌ NOT COVERED'}
- Disability Insurance: {'✅ Covered' if insurance_status.get('disability') else '❌ NOT COVERED'}
- Age: {user_data.get('age', 30)}
- Monthly Income: ₹{user_data.get('monthly_income', 0):,.0f}
- Insurance Score: {scores.get('insurance', 0):.1f}/10

## Knowledge Base Context
{context}

## Your Task
1. Identify the CRITICAL gaps in their coverage
2. Explain the financial risk of each gap (with real-world cost examples)
3. Suggest insurance premiums that fit WITHIN your budget of ₹{state.get('budget_allocations', {}).get('insurance', 0):,.0f}/month
4. Prioritize: Health > Life (if dependents) > Disability

Keep response under 350 words. Use markdown formatting.
"""

    response = invoke_with_retry(llm, [HumanMessage(content=prompt)], agent_name="Insurance Expert")
    advice = response.content

    recommendations = state.get("recommendations", {})
    recommendations["insurance"] = advice

    return {
        "recommendations": recommendations,
        "messages": [AIMessage(content=f"🛡️ **Insurance Expert Report**\n\n{advice}", name="internal")],
        **_advance_index(state),
    }


# ══════════════════════════════════════════════════════════════════
# NODE: Investment Agent (Investment Scout)
# ══════════════════════════════════════════════════════════════════
def investment_agent(state: dict) -> dict:
    """Check 100-Age equity allocation and retirement savings."""
    user_data = state.get("user_data", {})
    scores = state.get("scores", {})
    engine = FinancialEngine(user_data)

    context = retrieve_context(
        "investment_scout",
        "100-age rule equity allocation retirement savings fidelity milestones index funds"
    )

    ideal_eq = engine.get_ideal_equity_pct()
    current_eq = user_data.get("current_equity_pct", 50)

    llm = get_llm()
    prompt = f"""{COACH_PERSONA}

You are the **Investment Scout** — your mission is to optimize this user's wealth-building strategy.

{_budget_context(state, "investment")}

## User's Investment Profile
- Age: {user_data.get('age', 30)}
- Current Equity Allocation: {current_eq}%
- Ideal Equity (100-Age Rule): {ideal_eq}%
- Retirement Savings: ₹{user_data.get('retirement_savings', 0):,.0f}
- Annual Retirement Contribution: ₹{user_data.get('annual_retirement_contribution', 0):,.0f}
- Annual Income: ₹{user_data.get('annual_income', user_data.get('monthly_income', 0) * 12):,.0f}
- Investment Score: {scores.get('investment', 0):.1f}/10
- Retirement Score: {scores.get('retirement', 0):.1f}/10

## Knowledge Base Context
{context}

## Your Task
1. Compare their current vs. ideal allocation — recommend specific changes
2. Check against Fidelity milestones — are they on track?
3. Suggest monthly contribution WITHIN your budget of ₹{state.get('budget_allocations', {}).get('investment', 0):,.0f}/month
4. Recommend specific investment vehicles (three-fund portfolio)
5. If behind on retirement, calculate a catch-up plan within budget

Keep response under 400 words. Use markdown formatting.
"""

    response = invoke_with_retry(llm, [HumanMessage(content=prompt)], agent_name="Investment Scout")
    advice = response.content

    recommendations = state.get("recommendations", {})
    recommendations["investment"] = advice

    return {
        "recommendations": recommendations,
        "messages": [AIMessage(content=f"📈 **Investment Scout Report**\n\n{advice}", name="internal")],
        **_advance_index(state),
    }





# ══════════════════════════════════════════════════════════════════
# NODE: State Updater
# ══════════════════════════════════════════════════════════════════
def state_updater(state: dict) -> dict:
    """
    After ALL agents have run, estimate projected scores
    if the user follows every recommendation.
    """
    scores = state.get("scores", {})
    recommendations = state.get("recommendations", {})
    budget_allocations = state.get("budget_allocations", {})
    surplus = state.get("available_monthly_surplus", 0)

    projected = dict(scores)

    # Improvement is proportional to budget fraction (more money → bigger improvement)
    for agent_key, budget in budget_allocations.items():
        if agent_key not in recommendations:
            continue
        # Improvement potential: up to +4 for full budget, scaled linearly
        if surplus > 0:
            budget_fraction = budget / surplus
        else:
            budget_fraction = 0.2  # even with no surplus, free actions help a little

        improvement = budget_fraction * 4

        if agent_key == "savings":
            projected["emergency"] = min(10, scores.get("emergency", 0) + improvement + 1)
        elif agent_key == "debt":
            projected["debt"] = min(10, scores.get("debt", 0) + improvement + 1)
        elif agent_key == "insurance":
            projected["insurance"] = min(10, scores.get("insurance", 0) + improvement + 1)
        elif agent_key == "investment":
            projected["investment"] = min(10, scores.get("investment", 0) + improvement)
            projected["retirement"] = min(10, scores.get("retirement", 0) + improvement)

    # Round projections
    projected = {k: round(v, 1) for k, v in projected.items()}

    # Generate comparison chart
    comparison_path = create_comparison_chart(scores, projected)

    return {
        "projected_scores": projected,
    }


# ══════════════════════════════════════════════════════════════════
# NODE: Final Planner
# ══════════════════════════════════════════════════════════════════
# Ordered label map for display
AGENT_LABELS = {
    "savings":    "🏦 Savings Architect",
    "debt":       "🔥 Debt Shredder",
    "insurance":  "🛡️ Insurance Expert",
    "investment": "📈 Investment Scout",
}


def final_planner(state: dict) -> dict:
    """
    Synthesize ALL 4 agent recommendations into a prioritised Financial
    Flight Plan.  Uses the severity order and budget allocations to
    ensure the roadmap is realistic and within budget.
    """
    recommendations = state.get("recommendations", {})
    scores = state.get("scores", {})
    projected = state.get("projected_scores", scores)
    user_data = state.get("user_data", {})
    budget_allocations = state.get("budget_allocations", {})
    agent_order = state.get("agent_execution_order", DEFAULT_PRIORITY)
    surplus = state.get("available_monthly_surplus", 0)

    if not recommendations:
        plan = (
            "## 🎉 Financial Flight Plan\n\n"
            "**Congratulations!** All your financial health scores are in the green zone. "
            "Keep doing what you're doing!\n\n"
            "**Next Steps:**\n"
            "- Review your plan quarterly\n"
            "- Consider lifestyle inflation prevention\n"
            "- Look into estate planning if not done already\n"
        )
        return {
            "financial_plan": plan,
            "messages": [AIMessage(content=plan, name="internal")],
        }

    llm = get_llm(temperature=0.2)

    # ── Build structured digest of every agent's report ────────
    all_advice_sections = []
    for key in agent_order:
        label = AGENT_LABELS.get(key, key.title())
        advice = recommendations.get(key, "_No report available._")
        score_key = AGENT_SCORE_MAP.get(key, key)
        score_val = scores.get(score_key if key != "savings" else "emergency", "N/A")
        budget = budget_allocations.get(key, 0)
        all_advice_sections.append(
            f"### {label}  (Score: {score_val}/10 · Budget: ₹{budget:,.0f}/mo)\n{advice}"
        )

    all_advice = "\n\n---\n\n".join(all_advice_sections)

    # ── Budget summary table ───────────────────────────────────
    budget_table = "\n".join(
        f"- {AGENT_LABELS.get(k, k)}: ₹{budget_allocations.get(k, 0):,.0f}/mo"
        for k in agent_order
    )

    prompt = f"""{COACH_PERSONA}

You are the **Flight Plan Architect**.  You have received reports from ALL FOUR
specialist agents, run in severity order.  Your job is to merge their recommendations
into ONE cohesive, prioritised **Financial Flight Plan** — a clear step-by-step
roadmap the user can follow.

## CRITICAL BUDGET CONSTRAINT
- The user's total monthly surplus is **₹{surplus:,.0f}** (Income ₹{user_data.get('monthly_income', 0):,.0f} − Expenses ₹{user_data.get('essential_expenses', 0):,.0f}).
- The Brain has allocated this surplus as follows:
{budget_table}
- **The total of ALL recommendations MUST NOT exceed ₹{surplus:,.0f}/month.**
- If an agent recommended something outside its budget, override it.

## User Profile
- Age: {user_data.get('age', 30)}
- Monthly Income: ₹{user_data.get('monthly_income', 0):,.0f}
- Monthly Expenses: ₹{user_data.get('essential_expenses', 0):,.0f}
- Liquid Savings: ₹{user_data.get('liquid_savings', 0):,.0f}
- Total Debt: ₹{user_data.get('total_debt', 0):,.0f}

## Current Scores  →  Projected Scores
{chr(10).join(
    f"- {k.replace('_', ' ').title()}: {scores.get(k, 0):.1f}  →  {projected.get(k, 0):.1f}"
    for k in scores
)}

## Severity Order (most urgent first)
{' → '.join(a.replace('_', ' ').title() for a in agent_order)}

## ═══  All 4 Specialist Reports  ═══
{all_advice}

## ═══  Instructions  ═══
Using EVERY report above, create the Financial Flight Plan in Markdown:

1. **📋 Executive Summary** — 2-3 sentences covering the big picture
2. **🚨 Immediate Actions (This Week)** — Top 3-5 urgent steps from the
   most-severe agents
3. **📅 30-Day Game Plan** — Monthly actions with exact rupee amounts,
   totalling EXACTLY ₹{surplus:,.0f}/month
4. **🎯 90-Day Milestones** — Measurable targets for each dimension
5. **🏆 12-Month Vision** — Where the user will be across ALL dimensions
6. **💡 Monthly Budget Split** — A table showing:
   | Category | Amount | Purpose |
   with the total row = ₹{surplus:,.0f}
7. **📌 Priority Ranking** — Numbered list ranking all 4 areas from
   most-urgent to least-urgent with one-line justification

IMPORTANT RULES:
- All amounts must be in Indian Rupees (₹).
- Every bullet must include a specific rupee amount OR a concrete free action.
- Cross-reference agents (e.g., once debt is paid off, redirect to investing).
- Resolve any conflicts between agents.
- The GRAND TOTAL of monthly allocations MUST equal ₹{surplus:,.0f}. NO MORE.
"""

    response = invoke_with_retry(llm, [HumanMessage(content=prompt)], agent_name="Final Planner")
    plan = f"# ✈️ Your Financial Flight Plan\n\n{response.content}"

    return {
        "financial_plan": plan,
        "messages": [AIMessage(content=plan, name="internal")],
    }
