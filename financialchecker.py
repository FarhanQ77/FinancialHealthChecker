#############################
# Financial Health Checker
# with Advanced Enhancements
#############################

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime
import math

from prophet import Prophet
from prophet.plot import plot_plotly
from typing import Dict, List, Tuple


#############################
# Utility / Helper Functions
#############################

def load_csv_data(file) -> pd.DataFrame:
    """
    Load a CSV of historical transactions or monthly aggregates.
    Expected columns:
        - Date (YYYY-MM-DD or MM/DD/YYYY)
        - Income
        - Expenses
    """
    df = pd.read_csv(file, parse_dates=['Date'])
    df.sort_values(by='Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def prepare_time_series(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Convert a DataFrame with a 'Date' column and a numeric column
    into Prophet-compatible format: ds (datetime), y (value).
    """
    ts_df = df[['Date', column_name]].copy()
    ts_df.columns = ['ds', 'y']
    return ts_df


def forecast_expenses(df: pd.DataFrame, periods: int = 6) -> pd.DataFrame:
    """
    Use Prophet to forecast the 'y' column of a df (where df has columns ds, y).
    Returns the forecast DataFrame with yhat, yhat_lower, yhat_upper.
    """
    model = Prophet(yearly_seasonality=True, daily_seasonality=False)
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq='MS')  # monthly
    forecast = model.predict(future)
    return forecast, model


def debt_repayment_schedule(
        total_debt: float,
        monthly_payment: float,
        interest_rate: float = 0.18,
        strategy: str = "snowball"
) -> Tuple[int, float]:
    """
    Estimate months to pay off debt given a monthly payment strategy.

    For simplicity, we assume:
    - Single debt or combined debt amount
    - A nominal monthly interest rate derived from APR (interest_rate/12).
    - If strategy == 'snowball': same as avalanche in a single-debt scenario,
      but in multi-debt scenarios, you'd reorder by balance or rate.
    Returns:
      - months (int)
      - total_interest_paid (float)
    """
    monthly_interest_rate = interest_rate / 12.0
    balance = total_debt
    months = 0
    total_interest_paid = 0.0

    while balance > 0 and months < 600:  # safety limit
        # Add interest
        interest_for_month = balance * monthly_interest_rate
        total_interest_paid += interest_for_month
        balance += interest_for_month
        # Subtract payment
        balance -= monthly_payment
        months += 1
        if balance < 0:
            balance = 0

    return months, total_interest_paid


def generate_nudges(income: float, expenses: float, goals: Dict[str, float], debt: float):
    """
    Create basic 'nudges' or suggestions based on user data.
    Nudges can be more advanced using heuristics or ML-based
    classification in a real scenario.
    """
    suggestions = []
    surplus = income - expenses

    # Simple saving advice
    total_goal = sum(goals.values())
    if surplus <= 0:
        suggestions.append(
            "Your expenses match or exceed your income. Consider cutting discretionary spending or increasing income."
        )
    elif surplus < total_goal:
        suggestions.append(
            f"You have a surplus of {surplus:.2f}, but it's below your total savings goals ({total_goal:.2f}). "
            "Try reducing variable expenses to meet your goals."
        )
    else:
        suggestions.append("Great! You can fully meet your savings goals. Consider increasing them slightly.")

    # Debt advice
    if debt > 0:
        debt_ratio = (debt / income) * 100
        if debt_ratio > 40:
            suggestions.append(
                "High debt-to-income ratio detected. Prioritize paying down debt using avalanche or snowball method."
            )
        elif 20 < debt_ratio <= 40:
            suggestions.append("Your debt is moderate. Consider accelerating repayments to save on interest.")
        else:
            suggestions.append("Your debt ratio is fairly low, but consider extra payments to clear it faster.")

    # Behavioral suggestions
    suggestions.append("Automate your savings each month to reduce reliance on willpower.")
    suggestions.append("Try a 'cash envelope' or prepaid card system for entertainment/dining to avoid overspending.")

    return suggestions


def calculate_gamification_rewards(surplus: float, savings_goals_met: bool, debt_cleared: bool) -> List[str]:
    """
    Simple gamification: award 'badges' or achievements.
    """
    badges = []
    if surplus > 0:
        badges.append("Positive Income Badge")

    if savings_goals_met:
        badges.append("Savings Goal Achiever")

    if debt_cleared:
        badges.append("Debt-Free Champion")

    if not badges:
        badges.append("Keep Going! You're on the right track by analyzing your finances.")

    return badges


#############################
# Streamlit App Begins
#############################

def main():
    st.title("Financial Health Checker - Enhanced Edition")

    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio(
        "Choose a section",
        [
            "1) Data Input / CSV Upload",
            "2) Forecast & Analysis",
            "3) Debt & Savings",
            "4) Scenario Analysis",
            "5) Summary & Gamification"
        ]
    )

    # Persistent states
    if 'df' not in st.session_state:
        st.session_state.df = pd.DataFrame()
    if 'income' not in st.session_state:
        st.session_state.income = 0.0
    if 'fixed_expenses' not in st.session_state:
        st.session_state.fixed_expenses = 0.0
    if 'variable_expenses' not in st.session_state:
        st.session_state.variable_expenses = 0.0
    if 'debt' not in st.session_state:
        st.session_state.debt = 0.0
    if 'savings_buckets' not in st.session_state:
        st.session_state.savings_buckets = {"Emergency Fund": 0.0, "Vacation": 0.0, "Other": 0.0}
    if 'forecast_result' not in st.session_state:
        st.session_state.forecast_result = None
    if 'model' not in st.session_state:
        st.session_state.model = None

    if app_mode == "1) Data Input / CSV Upload":
        st.header("Step 1: Provide Your Financial Data")

        st.subheader("Option A: Manual Input")
        income = st.number_input("Monthly Income", min_value=0.0, step=100.0, value=st.session_state.income)
        fixed_expenses = st.number_input("Fixed Expenses (Rent, Utilities, etc.)", min_value=0.0, step=50.0,
                                         value=st.session_state.fixed_expenses)
        variable_expenses = st.number_input("Variable Expenses (Groceries, Dining, Entertainment)", min_value=0.0,
                                            step=50.0, value=st.session_state.variable_expenses)
        debt = st.number_input("Total Debt (All loans & credit)", min_value=0.0, step=100.0,
                               value=st.session_state.debt)

        with st.expander("Multiple Savings Buckets"):
            for bucket in st.session_state.savings_buckets:
                st.session_state.savings_buckets[bucket] = st.number_input(
                    f"Savings Goal: {bucket}",
                    min_value=0.0,
                    step=50.0,
                    value=st.session_state.savings_buckets[bucket]
                )

        if st.button("Save Manual Data"):
            st.session_state.income = income
            st.session_state.fixed_expenses = fixed_expenses
            st.session_state.variable_expenses = variable_expenses
            st.session_state.debt = debt
            st.success("Financial data saved to session.")

        st.subheader("Option B: Upload Monthly CSV")
        st.write("Expected columns: `Date`, `Income`, `Expenses`. We will aggregate them automatically.")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file is not None:
            st.session_state.df = load_csv_data(uploaded_file)
            st.success("CSV data loaded.")

        if not st.session_state.df.empty:
            st.write("Preview of the loaded data:")
            st.dataframe(st.session_state.df.head())

    elif app_mode == "2) Forecast & Analysis":
        st.header("Step 2: Forecast Your Expenses")

        # If user uploaded a CSV, we can use that for forecasting
        # else we can do a simple placeholder or require them to go back
        if st.session_state.df.empty:
            st.warning("No historical data found. Please upload CSV or proceed with manual scenario only.")
        else:
            df_expenses = prepare_time_series(st.session_state.df, 'Expenses')
            months_to_forecast = st.slider("Months to Forecast", 1, 24, 6)
            if st.button("Run Forecast"):
                forecast, model = forecast_expenses(df_expenses, periods=months_to_forecast)
                st.session_state.forecast_result = forecast
                st.session_state.model = model

            if st.session_state.forecast_result is not None:
                st.write("Forecasted Results (Head):")
                st.dataframe(st.session_state.forecast_result[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12))

                fig_forecast = plot_plotly(st.session_state.model, st.session_state.forecast_result)
                st.plotly_chart(fig_forecast, use_container_width=True)

    elif app_mode == "3) Debt & Savings":
        st.header("Step 3: Debt & Savings Strategies")

        st.write(f"**Current Income**: {st.session_state.income}")
        total_expenses = st.session_state.fixed_expenses + st.session_state.variable_expenses
        st.write(f"**Current Expenses**: {total_expenses}")
        st.write(f"**Current Debt**: {st.session_state.debt}")
        total_savings_goal = sum(st.session_state.savings_buckets.values())
        st.write(f"**Total Monthly Savings Goal**: {total_savings_goal}")

        st.subheader("Debt Repayment Strategy (Single Debt Approximation)")

        interest_rate_apr = st.slider("Annual Interest Rate (APR %) for Debt", 0.0, 35.0, 18.0)
        monthly_interest_rate = interest_rate_apr / 100.0
        surplus = st.session_state.income - total_expenses
        monthly_payment = st.number_input("Amount to allocate to debt monthly", 0.0, surplus if surplus > 0 else 1e6,
                                          min(surplus, 200.0))

        strategy = st.selectbox("Choose Strategy", ["snowball", "avalanche"])
        if st.button("Calculate Debt Schedule"):
            months_needed, total_interest_paid = debt_repayment_schedule(
                total_debt=st.session_state.debt,
                monthly_payment=monthly_payment,
                interest_rate=monthly_interest_rate,
                strategy=strategy
            )
            if months_needed >= 600:
                st.error("It seems the monthly payment is too small to clear your debt in a reasonable time.")
            else:
                st.success(f"**Months to Clear Debt**: {months_needed}")
                st.info(f"**Estimated Total Interest Paid**: {total_interest_paid:,.2f}")

        st.subheader("Savings Status")
        if surplus < total_savings_goal:
            st.warning(
                f"Your monthly surplus ({surplus:.2f}) is less than your total savings goal ({total_savings_goal:.2f}).")
        else:
            st.success(
                f"You can meet your total savings goal! Surplus after savings: {surplus - total_savings_goal:.2f}")

    elif app_mode == "4) Scenario Analysis":
        st.header("Step 4: What-If Scenarios")

        st.write("Explore how changes in spending or income affect your financial outlook.")
        income_scenario = st.slider("Adjust Monthly Income by (%)", -50, 100, 0)
        var_exp_scenario = st.slider("Adjust Variable Expenses by (%)", -50, 100, 0)
        debt_scenario = st.slider("Adjust Debt by (%)", -50, 100, 0)

        # Recalculate scenario
        new_income = st.session_state.income * (1 + (income_scenario / 100))
        new_variable_expenses = st.session_state.variable_expenses * (1 + (var_exp_scenario / 100))
        new_debt = st.session_state.debt * (1 + (debt_scenario / 100))

        scenario_expenses = st.session_state.fixed_expenses + new_variable_expenses
        scenario_surplus = new_income - scenario_expenses
        total_savings_goal = sum(st.session_state.savings_buckets.values())

        st.write(f"**Scenario Income**: {new_income:.2f}")
        st.write(f"**Scenario Expenses**: {scenario_expenses:.2f}")
        st.write(f"**Scenario Debt**: {new_debt:.2f}")
        st.write(f"**Scenario Surplus**: {scenario_surplus:.2f}")
        st.write(f"**Savings Goal**: {total_savings_goal:.2f}")

        if scenario_surplus >= total_savings_goal:
            st.success("You can still meet your savings goal under this scenario.")
        else:
            shortfall = total_savings_goal - scenario_surplus
            st.warning(
                f"You are short by {shortfall:.2f} to meet your savings goal. Consider adjusting expenses or debt repayment.")

    elif app_mode == "5) Summary & Gamification":
        st.header("Step 5: Summary, Nudges & Achievements")

        income = st.session_state.income
        expenses = st.session_state.fixed_expenses + st.session_state.variable_expenses
        goals = st.session_state.savings_buckets
        debt = st.session_state.debt

        # Nudges
        nudges = generate_nudges(income, expenses, goals, debt)
        st.subheader("Personalized Nudges")
        for i, suggestion in enumerate(nudges, start=1):
            st.write(f"{i}. {suggestion}")

        # Gamification
        st.subheader("Achievements Unlocked")
        surplus = income - expenses
        total_goal = sum(goals.values())
        savings_goals_met = (surplus >= total_goal)
        debt_cleared = (debt <= 0)
        badges = calculate_gamification_rewards(surplus, savings_goals_met, debt_cleared)

        for badge in badges:
            st.write(f"**- {badge}**")

        st.markdown("---")
        st.write("**Thank you for using the Enhanced Financial Health Checker!**")
        st.write("Feel free to navigate back to previous steps or adjust your data/scenarios to explore further.")


#############################
# Streamlit Runner
#############################
if __name__ == "__main__":
    main()
