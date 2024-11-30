import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.optimize import minimize
import wrds  # Import WRDS library

# Connect to WRDS before running Streamlit
@st.cache_resource
def connect_to_wrds():
    try:
        return wrds.Connection()
    except Exception as e:
        st.error("Error connecting to WRDS. Please check your credentials.")
        st.stop()

db = connect_to_wrds()

# UI: Title and User Inputs
st.title("Portfolio Optimization with Monte Carlo Simulation")
st.sidebar.header("Portfolio Parameters")

# Collect user inputs
tickers = st.sidebar.text_input(
    "Enter stock tickers separated by commas (e.g., AAPL, MSFT, AMZN)"
)
num_simulations = st.sidebar.number_input(
    "Number of simulations", min_value=100, max_value=10000, step=100, value=1000
)
risk_free_rate = st.sidebar.number_input(
    "Risk-free rate (%)", min_value=0.0, max_value=10.0, step=0.1, value=1.0
) / 100

# Data Retrieval from WRDS
def fetch_data_from_wrds(tickers, db):
    tickers = tickers.split(",")
    stock_data = pd.DataFrame()

    for ticker in tickers:
        ticker = ticker.strip()
        query = f"""
        SELECT date, ticker, prc AS close
        FROM crsp.dsf
        WHERE ticker = '{ticker}'
        AND date >= '2018-01-01'  -- Adjust date range as needed
        AND date <= CURRENT_DATE
        """
        try:
            df = db.raw_sql(query)
            if not df.empty:
                df.set_index("date", inplace=True)
                df.rename(columns={"close": ticker}, inplace=True)
                stock_data = pd.concat([stock_data, df[ticker]], axis=1)
            else:
                st.warning(f"No data found for ticker: {ticker}")
        except Exception as e:
            st.warning(f"Error fetching data for ticker {ticker}: {e}")

    return stock_data

# Check if user entered tickers
if tickers:
    data = fetch_data_from_wrds(tickers, db)
    if data.empty:
        st.error("No valid data found. Please check your tickers.")
    else:
        st.write("### Historical Stock Data", data.tail())

        # Calculate daily returns
        returns = data.pct_change().dropna()

        # Monte Carlo Simulation
        def monte_carlo_simulations(returns, num_simulations):
            np.random.seed(42)
            num_stocks = len(returns.columns)
            results = np.zeros((num_simulations, 3 + num_stocks))

            for i in range(num_simulations):
                weights = np.random.random(num_stocks)
                weights /= np.sum(weights)

                portfolio_return = np.sum(weights * returns.mean()) * 252
                portfolio_std_dev = np.sqrt(
                    np.dot(weights.T, np.dot(returns.cov() * 252, weights))
                )
                sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std_dev

                results[i, 0] = portfolio_return
                results[i, 1] = portfolio_std_dev
                results[i, 2] = sharpe_ratio
                results[i, 3:] = weights

            return results

        results = monte_carlo_simulations(returns, num_simulations)
        results_df = pd.DataFrame(
            results,
            columns=["Return", "Risk", "Sharpe Ratio"] + returns.columns.tolist(),
        )

        # Optimization - find the portfolio with the highest Sharpe ratio
        max_sharpe_idx = results_df["Sharpe Ratio"].idxmax()
        st.write("### Optimal Portfolio Allocation", results_df.iloc[max_sharpe_idx])

        # Visualization
        fig = px.scatter(
            results_df,
            x="Risk",
            y="Return",
            color="Sharpe Ratio",
            title="Monte Carlo Simulations: Risk vs Return",
        )
        fig.add_scatter(
            x=[results_df.loc[max_sharpe_idx, "Risk"]],
            y=[results_df.loc[max_sharpe_idx, "Return"]],
            mode="markers",
            marker=dict(color="red", size=10),
            name="Max Sharpe Ratio",
        )
        st.plotly_chart(fig)

        # Pie chart of optimal allocation
        optimal_weights = results_df.iloc[max_sharpe_idx, 3:]
        st.write("### Optimal Asset Allocation")
        fig = px.pie(
            values=optimal_weights.values,
            names=optimal_weights.index,
            title="Optimal Asset Allocation",
        )
        st.plotly_chart(fig)
