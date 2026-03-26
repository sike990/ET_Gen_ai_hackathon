"""
Anti-Gravity Financial Agentic System
LangGraph Workflow — StateGraph compilation with dispatcher-based chaining.

Flow:
  input_processor
       ↓
  brain (runs ONCE — scores, severity sort, budget allocation)
       ↓
  dispatcher ─→ agent₁ ─→ dispatcher ─→ agent₂ ─→ … ─→ dispatcher
       ↓ (all done)
  state_updater
       ↓
  final_planner
       ↓
      END
"""

from langgraph.graph import StateGraph, END
from state import AgentState
from agents import (
    input_processor,
    brain_orchestrator,
    agent_dispatcher,
    savings_agent,
    debt_agent,
    insurance_agent,
    investment_agent,
    state_updater,
    final_planner,
)


# ── Routing Functions ──────────────────────────────────────────────
def route_after_input(state: dict) -> str:
    """After input processing, go to brain if data is complete, else END."""
    if state.get("data_complete", False):
        return "brain"
    return END


def route_dispatcher(state: dict) -> str:
    """
    Read the active_agent set by the dispatcher.
    Routes to the matching specialist node, or to state_updater
    when all agents are done.
    """
    agent = state.get("active_agent", "done")
    route_map = {
        "savings":    "savings_agent",
        "debt":       "debt_agent",
        "insurance":  "insurance_agent",
        "investment": "investment_agent",
        "done":       "state_updater",
    }
    return route_map.get(agent, "state_updater")


# ── Build the Graph ────────────────────────────────────────────────
def build_graph() -> StateGraph:
    """
    Compile the Anti-Gravity Financial Agentic workflow.

    The key difference from the old design:
    • Brain runs ONCE (no loop).
    • A lightweight dispatcher reads agent_execution_order and
      current_agent_index to chain agents directly.
    • Each agent increments current_agent_index before returning,
      so the next dispatcher call picks the next agent.
    """
    workflow = StateGraph(AgentState)

    # ── Add Nodes ──────────────────────────────────────────────
    workflow.add_node("input_processor", input_processor)
    workflow.add_node("brain", brain_orchestrator)
    workflow.add_node("dispatcher", agent_dispatcher)
    workflow.add_node("savings_agent", savings_agent)
    workflow.add_node("debt_agent", debt_agent)
    workflow.add_node("insurance_agent", insurance_agent)
    workflow.add_node("investment_agent", investment_agent)
    workflow.add_node("state_updater", state_updater)
    workflow.add_node("final_planner", final_planner)

    # ── Entry Point ────────────────────────────────────────────
    workflow.set_entry_point("input_processor")

    # ── Edges ──────────────────────────────────────────────────

    # 1. Input → Brain (conditional: skip if data incomplete)
    workflow.add_conditional_edges(
        "input_processor",
        route_after_input,
        {"brain": "brain", END: END},
    )

    # 2. Brain → Dispatcher (always, brain only runs once)
    workflow.add_edge("brain", "dispatcher")

    # 3. Dispatcher → Specialist or State Updater (conditional)
    workflow.add_conditional_edges(
        "dispatcher",
        route_dispatcher,
        {
            "savings_agent":    "savings_agent",
            "debt_agent":       "debt_agent",
            "insurance_agent":  "insurance_agent",
            "investment_agent": "investment_agent",
            "state_updater":    "state_updater",
        },
    )

    # 4. Each specialist → Dispatcher (to continue the chain)
    for specialist in [
        "savings_agent", "debt_agent", "insurance_agent",
        "investment_agent",
    ]:
        workflow.add_edge(specialist, "dispatcher")

    # 5. State Updater → Final Planner
    workflow.add_edge("state_updater", "final_planner")

    # 6. Final Planner → END
    workflow.add_edge("final_planner", END)

    # ── Compile ────────────────────────────────────────────────
    compiled = workflow.compile()
    return compiled


# ── Convenience ────────────────────────────────────────────────────
def run_graph(user_data: dict) -> dict:
    """
    Execute the full workflow with the given user data.
    Returns the final state dict.
    """
    graph = build_graph()
    initial_state = {
        "user_data": user_data,
        "data_complete": False,
        "scores": {},
        "projected_scores": {},
        "agent_execution_order": [],
        "budget_allocations": {},
        "available_monthly_surplus": 0,
        "current_agent_index": 0,
        "messages": [],
        "recommendations": {},
        "spider_chart_path": None,
        "financial_plan": None,
    }
    final_state = graph.invoke(initial_state)
    return final_state


# ── CLI Test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = {
        "monthly_income": 5000,
        "essential_expenses": 3500,
        "total_debt": 15000,
        "monthly_debt_payment": 500,
        "debt_interest_rates": [22.0, 5.0],
        "debt_names": ["Credit Card", "Car Loan"],
        "debt_balances": [8000, 7000],
        "liquid_savings": 2000,
        "retirement_savings": 25000,
        "annual_retirement_contribution": 3000,
        "age": 32,
        "insurance_status": {"health": True, "life": False, "disability": False},
        "tax_advantaged_contributions": 3000,
        "employer_match_pct": 3,
        "current_equity_pct": 80,
        "annual_income": 60000,
    }
    print("🚀 Running Anti-Gravity Financial System...")
    result = run_graph(sample)
    print("\n" + "=" * 60)
    print(result.get("financial_plan", "No plan generated."))
