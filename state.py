"""
Anti-Gravity Financial Agentic System
State Definition — AgentState TypedDict for LangGraph
"""

from typing import TypedDict, Optional, Annotated
from langgraph.graph.message import add_messages


class UserData(TypedDict, total=False):
    """User's raw financial data collected from the Streamlit form."""
    monthly_income: float
    essential_expenses: float
    total_debt: float
    monthly_debt_payment: float
    debt_interest_rates: list[float]  # list of rates for each debt
    debt_names: list[str]             # corresponding debt names
    debt_balances: list[float]        # corresponding balances
    liquid_savings: float
    retirement_savings: float
    annual_retirement_contribution: float
    age: int
    insurance_status: dict  # {"health": bool, "life": bool, "disability": bool}
    tax_advantaged_contributions: float  # 401k / IRA annual
    employer_match_pct: float
    current_equity_pct: float  # current portfolio equity allocation %
    annual_income: float


class Scores(TypedDict, total=False):
    """Five financial health scores, each 0-10."""
    emergency: float
    debt: float
    insurance: float
    retirement: float
    investment: float


class AgentState(TypedDict, total=False):
    """
    Central state object passed through the LangGraph workflow.
    Every node reads from and writes to this state.
    """
    # --- User data ---
    user_data: UserData
    data_complete: bool  # True if all required fields are present

    # --- Scores ---
    scores: Scores
    projected_scores: Scores  # scores *after* acting on recommendations

    # --- Brain decisions (set ONCE, read by dispatcher) ---
    agent_execution_order: list[str]   # severity-sorted: ["debt","savings","insurance",...]
    budget_allocations: dict           # {"debt": 800, "savings": 400, ...}
    available_monthly_surplus: float   # income - essential_expenses
    current_agent_index: int           # which agent in the list to run next

    # --- Conversation ---
    messages: Annotated[list, add_messages]

    # --- Agent outputs ---
    recommendations: dict  # { agent_name: "advice markdown" }

    # --- Artifacts ---
    spider_chart_path: Optional[str]
    financial_plan: Optional[str]  # final Markdown flight plan
