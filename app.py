"""
Anti-Gravity Financial Agentic System
Streamlit Frontend — Dual-mode interface (Sidebar Form + Main Chat)
"""

import os
import re
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from graph import run_graph, build_graph
from scoring_engine import FinancialEngine
from visualization import create_spider_chart, create_comparison_chart
from rag_setup import ingest_documents
from langchain_core.messages import HumanMessage, AIMessage
from database import init_db, create_user, authenticate_user, save_user_profile, get_user_profile

# ── Page Config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Anti-Gravity Financial System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0D1117 0%, #161B22 50%, #0D1117 100%);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #161B22 0%, #0D1117 100%);
        border-right: 1px solid #21262D;
    }

    /* Header styling */
    .main-header {
        text-align: center;
        padding: 1rem 0 2rem 0;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #00E5FF 0%, #00BCD4 50%, #4CAF50 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: #8B949E;
        font-size: 1rem;
    }

    /* Score cards */
    .score-card {
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .score-red { background: rgba(255,75,75,0.15); border: 1px solid #FF4B4B; }
    .score-yellow { background: rgba(255,235,59,0.15); border: 1px solid #FFEB3B; }
    .score-green { background: rgba(76,175,80,0.15); border: 1px solid #4CAF50; }
    .score-value { font-size: 2rem; font-weight: 700; }
    .score-label { font-size: 0.85rem; color: #8B949E; text-transform: uppercase; }

    /* Chat messages */
    .chat-msg {
        padding: 1rem 1.2rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border: 1px solid #21262D;
        background: #161B22;
        color: #C9D1D9;
    }
    .chat-msg-user {
        background: #1A3A5C;
        border-color: #1F6FEB;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #00BCD4 0%, #00E5FF 100%);
        color: #0D1117;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        box-shadow: 0 0 20px rgba(0, 229, 255, 0.3);
        transform: translateY(-1px);
    }

    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #21262D, transparent);
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Session State Init ─────────────────────────────────────────────
if "graph_result" not in st.session_state:
    st.session_state.graph_result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "user_data" not in st.session_state:
    st.session_state.user_data = None
if "rag_initialized" not in st.session_state:
    st.session_state.rag_initialized = False
if "onboarding_shown" not in st.session_state:
    st.session_state.onboarding_shown = False
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    init_db()
if "username" not in st.session_state:
    st.session_state.username = None
if "user_id" not in st.session_state:
    st.session_state.user_id = None

# ── Show onboarding message on first load ──────────────────────────
if not st.session_state.onboarding_shown:
    st.session_state.onboarding_shown = True
    welcome_msg = """👋 **Welcome to your Financial Command Center!**

Before you hit the **🚀 Analyze** button, let me explain what each field in the sidebar means — no jargon, I promise!

---

### 💰 Income & Expenses
- **Monthly Income** — Your total salary/earnings that hit your bank account every month (after tax, i.e., in-hand salary)
- **Monthly Essential Expenses** — Rent, groceries, electricity, phone bill, EMIs, transport — basically everything you MUST pay every month to survive

### 🏦 Savings
- **Liquid Savings / Emergency Fund** — Cash you can access RIGHT NOW — savings account, FD that you can break, cash at home. NOT investments.
- **Retirement Savings (PF/NPS/PPF)** — Total amount sitting in your PF, NPS, PPF, or any retirement-specific account
- **Annual Retirement Contribution** — How much goes into these retirement accounts per YEAR (your PF contribution + employer's)

### 🔥 Debt
- Add each loan/credit card separately with its **balance** (how much you still owe), **interest rate** (the APR%), and **monthly payment** (your current EMI)

### 🛡️ Insurance
- Simply tick which insurance you currently have — health, life, or disability

### 📊 Age & Investments
- **Equity Allocation %** — If you invest in mutual funds or stocks, what % of your total investments are in equity (stocks/equity MFs)? If unsure, leave at 50%
- **Annual Tax-Advantaged Contributions** — How much you put into 80C instruments per year (ELSS, PPF, NPS, etc.)
- **Employer Match %** — If your company matches your PF contribution, what % do they match? (Most companies match 12% of basic)

---

📝 *Fill in what you know, leave defaults for what you don't — the AI will work with whatever you provide!*"""
    st.session_state.chat_history.append(("assistant", welcome_msg))


# ── Ensure RAG is initialized ─────────────────────────────────────
def init_rag():
    if not st.session_state.rag_initialized:
        with st.spinner("🔄 Initializing knowledge base..."):
            try:
                ingest_documents()
                st.session_state.rag_initialized = True
            except Exception as e:
                st.error(f"RAG initialization failed: {e}")


# ══════════════════════════════════════════════════════════════════
# SIDEBAR: Financial Data Form
# ══════════════════════════════════════════════════════════════════
if not st.session_state.logged_in:
    st.markdown("<h2 style='text-align: center; color: #00E5FF;'>🔐 Anti-Gravity Login</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Please login or register to access your financial profiles.</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            log_user = st.text_input("Username", key="log_user")
            log_pass = st.text_input("Password", type="password", key="log_pass")
            if st.button("Login", type="primary", use_container_width=True, key="log_btn"):
                if log_user and log_pass:
                    success, uid = authenticate_user(log_user, log_pass)
                    if success:
                        st.session_state.logged_in = True
                        st.session_state.username = log_user
                        st.session_state.user_id = uid
                        
                        profile = get_user_profile(uid)
                        if profile:
                            st.session_state.user_data = profile
                            
                        st.rerun()
                    else:
                        st.error("Invalid credentials.")
                else:
                    st.warning("Please fill all fields.")
                    
        with tab2:
            reg_user = st.text_input("New Username", key="reg_user")
            reg_pass = st.text_input("New Password", type="password", key="reg_pass")
            reg_conf = st.text_input("Confirm Password", type="password", key="reg_conf")
            if st.button("Register", type="primary", use_container_width=True, key="reg_btn"):
                if reg_user and reg_pass and reg_conf:
                    if reg_pass == reg_conf:
                        success, msg = create_user(reg_user, reg_pass)
                        if success:
                            st.success("Registered successfully! Please login from the Login tab.")
                        else:
                            st.error(msg)
                    else:
                        st.warning("Passwords do not match!")
                else:
                    st.warning("Please fill all fields.")
                    
    st.stop()

# ══════════════════════════════════════════════════════════════════
# SIDEBAR: Financial Data Form
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"**👤 Logged in as: {st.session_state.username}**")
    if st.button("Logout", key="logout_btn", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.user_id = None
        st.session_state.user_data = None
        st.session_state.graph_result = None
        st.session_state.chat_history = []
        st.rerun()
        
    st.markdown("---")
    st.markdown("## 🚀 Your Financial Profile")
    st.markdown("---")
    
    saved_profile = st.session_state.user_data or {}

    # --- Income & Expenses ---
    st.markdown("### 💰 Income & Expenses")
    monthly_income = st.number_input(
        "Monthly Income (₹)", min_value=0, value=int(saved_profile.get("monthly_income", 50000)), step=1000, key="income"
    )
    essential_expenses = st.number_input(
        "Monthly Essential Expenses (₹)", min_value=0, value=int(saved_profile.get("essential_expenses", 30000)), step=1000, key="expenses"
    )

    st.markdown("---")

    # --- Savings ---
    st.markdown("### 🏦 Savings")
    liquid_savings = st.number_input(
        "Liquid Savings / Emergency Fund (₹)", min_value=0, value=int(saved_profile.get("liquid_savings", 100000)), step=5000, key="savings"
    )
    retirement_savings = st.number_input(
        "Retirement Savings (PF/NPS/PPF) (₹)", min_value=0, value=int(saved_profile.get("retirement_savings", 500000)), step=10000, key="retirement"
    )
    annual_retirement_contribution = st.number_input(
        "Annual Retirement Contribution (₹)", min_value=0, value=int(saved_profile.get("annual_retirement_contribution", 60000)), step=5000, key="ret_contrib"
    )

    st.markdown("---")

    # --- Debt ---
    st.markdown("### 🔥 Debt")
    num_debts = st.number_input("Number of Debts", min_value=0, max_value=10, value=2, step=1)

    debt_names = []
    debt_balances = []
    debt_rates = []
    total_monthly_payment = 0

    for i in range(int(num_debts)):
        with st.expander(f"Debt #{i+1}", expanded=(i == 0)):
            name = st.text_input(f"Name", value=f"Debt {i+1}", key=f"dname_{i}")
            balance = st.number_input(f"Balance (₹)", min_value=0, value=50000, step=5000, key=f"dbal_{i}")
            rate = st.number_input(f"Interest Rate (%)", min_value=0.0, value=15.0, step=0.5, key=f"drate_{i}")
            payment = st.number_input(f"Monthly Payment (₹)", min_value=0, value=5000, step=500, key=f"dpay_{i}")
            debt_names.append(name)
            debt_balances.append(balance)
            debt_rates.append(rate)
            total_monthly_payment += payment

    st.markdown("---")

    # --- Insurance ---
    st.markdown("### 🛡️ Insurance")
    has_health = st.checkbox("Health Insurance", value=True, key="health_ins")
    has_life = st.checkbox("Life Insurance", value=False, key="life_ins")
    has_disability = st.checkbox("Disability Insurance", value=False, key="disability_ins")

    st.markdown("---")

    # --- Age & Investment ---
    st.markdown("### 📊 Age & Investments")
    age = st.number_input("Age", min_value=18, max_value=80, value=int(saved_profile.get("age", 30)), step=1, key="age")
    current_equity_pct = st.slider("Current Equity Allocation (%)", 0, 100, int(saved_profile.get("current_equity_pct", 70)), key="equity")
    tax_contributions = st.number_input(
        "Annual Tax-Advantaged Contributions (₹)", min_value=0, value=int(saved_profile.get("tax_advantaged_contributions", 150000)), step=10000, key="tax_contrib"
    )
    employer_match = st.number_input(
        "Employer Match (%)", min_value=0.0, max_value=20.0, value=float(saved_profile.get("employer_match_pct", 3.0)), step=0.5, key="match"
    )

    st.markdown("---")

    # --- Submit ---
    analyze_btn = st.button("🚀 Analyze My Finances", use_container_width=True, type="primary")
    save_btn = st.button("💾 Save Profile Configuration", use_container_width=True)

    if save_btn:
        user_data_to_save = {
            "monthly_income": monthly_income,
            "essential_expenses": essential_expenses,
            "liquid_savings": liquid_savings,
            "retirement_savings": retirement_savings,
            "annual_retirement_contribution": annual_retirement_contribution,
            "total_debt": sum(debt_balances),
            "monthly_debt_payment": total_monthly_payment,
            "debt_interest_rates": debt_rates,
            "debt_names": debt_names,
            "debt_balances": debt_balances,
            "age": age,
            "insurance_status": {
                "health": has_health,
                "life": has_life,
                "disability": has_disability,
            },
            "tax_advantaged_contributions": tax_contributions,
            "employer_match_pct": employer_match,
            "current_equity_pct": current_equity_pct,
            "annual_income": monthly_income * 12,
        }
        
        success = save_user_profile(st.session_state.user_id, user_data_to_save)
        if success:
            st.session_state.user_data = user_data_to_save
            st.success("✅ Profile saved!")
        else:
            st.error("❌ Failed to save profile.")


# ══════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h1>🚀 Anti-Gravity Financial System</h1>
    <p>AI-Powered Multi-Agent Financial Advisor</p>
</div>
""", unsafe_allow_html=True)


# ── Handle form submission ─────────────────────────────────────────
if analyze_btn:
    init_rag()

    user_data = {
        "monthly_income": monthly_income,
        "essential_expenses": essential_expenses,
        "liquid_savings": liquid_savings,
        "retirement_savings": retirement_savings,
        "annual_retirement_contribution": annual_retirement_contribution,
        "total_debt": sum(debt_balances),
        "monthly_debt_payment": total_monthly_payment,
        "debt_interest_rates": debt_rates,
        "debt_names": debt_names,
        "debt_balances": debt_balances,
        "age": age,
        "insurance_status": {
            "health": has_health,
            "life": has_life,
            "disability": has_disability,
        },
        "tax_advantaged_contributions": tax_contributions,
        "employer_match_pct": employer_match,
        "current_equity_pct": current_equity_pct,
        "annual_income": monthly_income * 12,
    }

    st.session_state.user_data = user_data

    with st.spinner("🧠 Running Anti-Gravity Financial Brain..."):
        try:
            result = run_graph(user_data)
            st.session_state.graph_result = result

            # Extract chat messages from the result
            messages = result.get("messages", [])
            st.session_state.chat_history = []
            for msg in messages:
                if isinstance(msg, AIMessage):
                    st.session_state.chat_history.append(("assistant", msg.content))
                elif isinstance(msg, HumanMessage):
                    st.session_state.chat_history.append(("user", msg.content))

        except Exception as e:
            st.error(f"❌ Error running the financial system: {e}")
            st.exception(e)

# ── Display Results ────────────────────────────────────────────────
result = st.session_state.graph_result

if result:
    scores = result.get("scores", {})
    projected = result.get("projected_scores", {})

    # ── Score Cards ────────────────────────────────────────────
    st.markdown("### 📊 Financial Health Dashboard")
    cols = st.columns(5)
    score_items = list(scores.items())

    for idx, (key, value) in enumerate(score_items):
        with cols[idx % 5]:
            css_class = "score-red" if value < 4 else ("score-yellow" if value < 8 else "score-green")
            color = "#FF4B4B" if value < 4 else ("#FFEB3B" if value < 8 else "#4CAF50")
            st.markdown(f"""
                <div class="score-card {css_class}">
                    <div class="score-value" style="color: {color}">{value:.1f}</div>
                    <div class="score-label">{key.replace('_', ' ')}</div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Budget Allocation ──────────────────────────────────────
    budget_alloc = result.get("budget_allocations", {})
    agent_order = result.get("agent_execution_order", [])
    surplus = result.get("available_monthly_surplus", 0)

    if budget_alloc and surplus > 0:
        st.markdown(f"### 💰 Budget Allocation (₹{surplus:,.0f}/mo surplus)")
        budget_cols = st.columns(len(budget_alloc) if budget_alloc else 1)
        for idx, agent_key in enumerate(agent_order):
            amount = budget_alloc.get(agent_key, 0)
            pct = (amount / surplus * 100) if surplus > 0 else 0
            with budget_cols[idx % len(budget_cols)]:
                st.metric(
                    label=agent_key.replace('_', ' ').title(),
                    value=f"₹{amount:,.0f}",
                    delta=f"{pct:.0f}% of surplus",
                )
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Charts ─────────────────────────────────────────────────
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("#### Current Health Radar")
        chart_path = result.get("spider_chart_path")
        if chart_path and os.path.exists(chart_path):
            st.image(chart_path, use_container_width=True)
        else:
            # Regenerate
            chart_path = create_spider_chart(scores)
            st.image(chart_path, use_container_width=True)

    with chart_col2:
        st.markdown("#### Current vs. Projected")
        if projected and projected != scores:
            comparison_path = create_comparison_chart(scores, projected)
            st.image(comparison_path, use_container_width=True)
        else:
            st.info("📈 Projected scores will appear once the agents provide recommendations.")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Financial Flight Plan ──────────────────────────────────
    plan = result.get("financial_plan")
    if plan:
        with st.expander("✈️ **Your Financial Flight Plan**", expanded=True):
            st.markdown(plan)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


elif not analyze_btn:
    # ── Welcome Screen ─────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding: 3rem 2rem;">
        <h2 style="color:#C9D1D9; margin-bottom:1rem;">Welcome to Your Financial Command Center</h2>
        <p style="color:#8B949E; font-size:1.1rem; max-width:600px; margin:0 auto;">
            Fill out the sidebar form with your financial details, then hit
            <strong style="color:#00E5FF;">🚀 Analyze My Finances</strong>
            to unleash the AI agents.
        </p>
        <div style="margin-top:2rem; padding:1.5rem; background:#161B22; border-radius:12px;
                    border:1px solid #21262D; max-width:500px; margin-left:auto; margin-right:auto;">
            <p style="color:#00E5FF; font-weight:600; margin-bottom:0.8rem;">🧠 How it works</p>
            <p style="color:#8B949E; font-size:0.9rem; text-align:left; line-height:1.8;">
                1️⃣ <strong>Input</strong> → You provide your financial data<br>
                2️⃣ <strong>Score</strong> → The engine calculates 6 health scores<br>
                3️⃣ <strong>Budget</strong> → The Brain allocates your surplus by severity<br>
                4️⃣ <strong>Chain</strong> → Agents run worst-score-first, each within budget<br>
                5️⃣ <strong>Plan</strong> → Everything is synthesized into your Flight Plan
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────
# CHAT SECTION — always visible, outside the result/welcome blocks
# ──────────────────────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("### 💬 Agent Conversation")
for role, content in st.session_state.chat_history:
    if role == "assistant":
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(content)
    else:
        with st.chat_message("user", avatar="👤"):
            st.markdown(content)


# ── Chat Input (for updates) ──────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

chat_input = st.chat_input("💬 Tell me about a financial change (e.g., 'I got a raise to ₹80,000/month')...")

if chat_input and result:
    st.session_state.chat_history.append(("user", chat_input))

    # Check if the message contains new financial data
    updated = False
    user_data = st.session_state.user_data or {}

    # Simple pattern matching for common updates
    income_match = re.search(r'(?:income|raise|salary|earn)[^\d]*[₹$]?([\d,]+)', chat_input, re.I)
    savings_match = re.search(r'(?:saved|savings|emergency)[^\d]*[₹$]?([\d,]+)', chat_input, re.I)
    debt_match = re.search(r'(?:paid off|paid|debt)[^\d]*[₹$]?([\d,]+)', chat_input, re.I)

    if income_match:
        new_income = float(income_match.group(1).replace(",", ""))
        user_data["monthly_income"] = new_income
        user_data["annual_income"] = new_income * 12
        updated = True
        st.session_state.chat_history.append(
            ("assistant", f"📝 Updated your monthly income to **₹{new_income:,.0f}**. Re-running analysis...")
        )

    if savings_match:
        new_savings = float(savings_match.group(1).replace(",", ""))
        user_data["liquid_savings"] = new_savings
        updated = True
        st.session_state.chat_history.append(
            ("assistant", f"📝 Updated your savings to **₹{new_savings:,.0f}**. Re-running analysis...")
        )

    if debt_match:
        paid_amount = float(debt_match.group(1).replace(",", ""))
        user_data["total_debt"] = max(0, user_data.get("total_debt", 0) - paid_amount)
        updated = True
        st.session_state.chat_history.append(
            ("assistant", f"📝 Reduced your debt by **₹{paid_amount:,.0f}**. Re-running analysis...")
        )

    if updated:
        st.session_state.user_data = user_data
        with st.spinner("🔄 Re-analyzing with updated data..."):
            try:
                new_result = run_graph(user_data)
                st.session_state.graph_result = new_result

                messages = new_result.get("messages", [])
                for msg in messages:
                    if isinstance(msg, AIMessage):
                        st.session_state.chat_history.append(("assistant", msg.content))

            except Exception as e:
                st.session_state.chat_history.append(("assistant", f"❌ Error: {e}"))
    else:
        # Just a general chat message — respond conversationally
        from agents import get_llm, COACH_PERSONA, invoke_with_retry
        llm = get_llm()
        scores = result.get("scores", {})
        prompt = f"""{COACH_PERSONA}

The user has already been analyzed. Their current scores are:
{chr(10).join(f"- {k}: {v:.1f}/10" for k, v in scores.items())}

The user says: "{chat_input}"

Respond helpfully. If they're asking about their finances, reference their scores.
If they're providing new information, ask them to update the sidebar form.
Keep response under 200 words.
"""
        response = invoke_with_retry(llm, [HumanMessage(content=prompt)], agent_name="Chat Interface")
        st.session_state.chat_history.append(("assistant", response.content))

    st.rerun()

elif chat_input and not result:
    st.session_state.chat_history.append(("user", chat_input))
    st.session_state.chat_history.append(
        ("assistant", "👋 Please fill out the sidebar form and click **🚀 Analyze My Finances** first!")
    )
    st.rerun()
