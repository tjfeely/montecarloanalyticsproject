Portfolio Optimization Using Monte Carlo Simulations

Overview
This project demonstrates portfolio optimization using Monte Carlo simulations to explore different portfolio allocations and maximize the Sharpe ratio. The project is implemented in Python with an interactive user interface built using Streamlit.

Key Features
Dynamic User Inputs:
Specify stock tickers to analyze.
Define the number of Monte Carlo simulations.
Set the risk-free rate for Sharpe ratio calculations.

Automated Data Retrieval:
Fetch historical stock prices from Yahoo Finance via the yfinance library (5 years).

Monte Carlo Simulations:
Simulate thousands of portfolio weight combinations.
Compute portfolio return, risk, and Sharpe ratio for each simulation.

Portfolio Optimization:
Identify the portfolio allocation that maximizes the Sharpe ratio.

Visualizations:
Scatter plot to visualize simulated portfolios.
Pie chart for the optimal portfolio allocation.
Technologies Used

Python Libraries:
Streamlit: Build the interactive web interface.
yfinance: Fetch historical stock price data.
Pandas: Data manipulation and analysis.
Numpy: Numerical computations.
Matplotlib and Plotly: Create visualizations.
Scipy.optimize: Optimize portfolio weights.

Monte Carlo Simulation:
Randomly generates portfolio weights.
Computes portfolio risk, return, and Sharpe ratio.

How It Works

User Input:
Enter stock tickers, the number of simulations, and the risk-free rate through a simple web interface.
Data Retrieval:
Fetch 5 years of historical stock prices for the specified tickers.

Monte Carlo Simulations:
Simulate different portfolio weight allocations.
Calculate risk (standard deviation), return, and Sharpe ratio for each simulated portfolio.

Optimization:
Identify the portfolio with the highest Sharpe ratio (optimal portfolio).

Visualization:
Scatter plot of portfolio risk vs. return.
Pie chart of the optimal portfolio allocation.
