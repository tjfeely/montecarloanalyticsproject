import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.optimize import minimize

# UI: Title and User Inputs
st.title("Portfolio Optimization with Monte Carlo Simulation")
st.sidebar.header("Portfolio Parameters")

# Collect user inputs
tickers = st.sidebar.text_input("Enter stock tickers separated by commas (e.g., AAPL, MSFT, AMZN)")
num_simulations = st.sidebar.number_input("Number of simulations", min_value=100, max_value=10000, step=100, value=1000)
risk_free_rate = st.sidebar.number_input("Risk-free rate (%)", min_value=0.0, max_value=10.0, step=0.1, value=1.0) / 100

# Data Retrieval
def fetch_data(tickers):
    stock_data = {}
    for ticker in tickers.split(','):
        stock_data[ticker.strip()] = yf.download(ticker.strip(), period="5y")['Adj Close']
    return pd.DataFrame(stock_data)

# Check if user entered tickers
if tickers:
    data = fetch_data(tickers)
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
            portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std_dev
            
            results[i, :3] = [portfolio_return, portfolio_std_dev, sharpe_ratio]
            results[i, 3:] = weights
        
        return results

    results = monte_carlo_simulations(returns, num_simulations)
    results_df = pd.DataFrame(results, columns=['Return', 'Risk', 'Sharpe Ratio'] + returns.columns.tolist())

    # Optimization - find the portfolio with the highest Sharpe ratio
    max_sharpe_idx = results_df['Sharpe Ratio'].idxmax()
    st.write("### Optimal Portfolio Allocation", results_df.iloc[max_sharpe_idx])

    # Visualizations
    fig = px.scatter(results_df, x='Risk', y='Return', color='Sharpe Ratio', title="Monte Carlo Simulations: Risk vs Return")
    fig.add_scatter(x=[results_df.loc[max_sharpe_idx, 'Risk']], y=[results_df.loc[max_sharpe_idx, 'Return']],
                    mode='markers', marker=dict(color='red', size=10), name="Max Sharpe Ratio")
    st.plotly_chart(fig)

    # Pie chart of optimal allocation
    optimal_weights = results_df.iloc[max_sharpe_idx, 3:]
    st.write("### Optimal Asset Allocation")
    fig = px.pie(values=optimal_weights, names=optimal_weights.index)
    st.plotly_chart(fig)