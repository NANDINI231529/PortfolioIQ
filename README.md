# PortfolioIQ

**Portfolio Risk and Return Analysis** is an educational Statistical Machine Learning web application for studying historical Indian equity returns, portfolio risk, random long-only portfolios, and constrained portfolio optimization.

## Features

- Local sign-up and login with salted PBKDF2-HMAC-SHA256 password hashes; no default credentials.
- Protected Streamlit dashboard with logout and session-based authentication.
- Yahoo Finance adjusted-price downloads for Reliance Industries, TCS, Infosys, and HDFC Bank.
- Historical price and daily-return views, stock-level metrics, and a daily-return correlation heatmap.
- Annualized arithmetic and geometric returns, sample variance, volatility, covariance, and correlation.
- Reproducible Monte Carlo-style long-only portfolio simulations with configurable count and seed.
- SciPy SLSQP minimum-volatility and maximum-Sharpe optimizations under fully invested, long-only constraints.
- CSV downloads for prices, returns, metrics, matrices, simulations, and optimizer results.
- Offline unit tests that use explicitly synthetic test fixtures and never substitute synthetic observations for live data.

## Technology stack

Python 3.12, Streamlit, Pandas, NumPy, Matplotlib, Seaborn, yfinance, SciPy, SQLite, and standard-library authentication and testing tools.

## Project structure

```text
PortfolioIQ/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/config.toml
├── assets/README.md
├── authentication/
│   ├── __init__.py
│   └── auth.py
├── analytics/
│   ├── __init__.py
│   ├── market_data.py
│   ├── portfolio.py
│   ├── returns.py
│   └── validation.py
├── components/
│   ├── __init__.py
│   └── ui.py
├── database/
│   ├── __init__.py
│   └── db.py
├── data/                 # Created/used at runtime; local account database is excluded from ZIP
└── tests/
    ├── __init__.py
    ├── test_auth.py
    ├── test_dashboard.py
    ├── test_database.py
    ├── test_exports.py
    ├── test_market_data.py
    ├── test_portfolio.py
    ├── test_returns.py
    └── test_validation.py
```

## Requirements

- Python 3.12 is the target interpreter.
- A network connection is needed to download Yahoo Finance data and install dependencies.
- No API keys are required.
- Python 3.13 may work with compatible current dependencies, but this project was specified for Python 3.12 and other interpreter versions are not claimed as verified.

## Windows Command Prompt installation and launch

Download and extract `PortfolioIQ.zip` to the directory where you want to keep the project. In Command Prompt, run:

```bat
REM If PortfolioIQ.zip is in the current directory:
powershell -NoProfile -Command "Expand-Archive -LiteralPath PortfolioIQ.zip -DestinationPath . -Force"

REM Step 1: Navigate to the project directory
cd PortfolioIQ

REM Step 2: Create a virtual environment
py -3.12 -m venv .venv

REM Step 3: Activate the environment
.venv\Scripts\activate

REM Step 4: Upgrade pip
python -m pip install --upgrade pip

REM Step 5: Install dependencies
pip install -r requirements.txt

REM Step 6: Run tests
python -m unittest discover -s tests -v

REM Step 7: Launch the dashboard
python -m streamlit run app.py
```

If `py -3.12` is unavailable, check installed interpreters with `py --list`. Install Python 3.12 from python.org if needed, then use the explicit interpreter path, for example:

```bat
"C:\Path\To\Python312\python.exe" -m venv .venv
```

Activate the resulting environment with `.venv\Scripts\activate` and continue with the `python -m pip` commands above. Streamlit starts a local application server and normally opens the dashboard in your browser.

## Accounts and dashboard use

1. Open PortfolioIQ and choose **Create account**.
2. Enter your name, email, and a password of at least eight characters.
3. Return to Login and sign in with those credentials.
4. Choose one or more stocks, the analysis period, simulation count, random seed, and annual risk-free rate from the sidebar.
5. Review the stock metrics, historical chart, return correlations, portfolio simulations, and optimized allocations.
6. Use the CSV buttons to export computed analysis. Exports contain financial results only and no account details.
7. Use **Log out** to clear authenticated identity from the Streamlit session.

The SQLite account database is initialized automatically at `data/portfolioiq.sqlite3` the first time the app runs. It is local to the machine that runs Streamlit and is intentionally not prepopulated or included in the ZIP.

## Financial methodology

Let `Pₜ` be the adjusted closing price at observation `t`, and `rₜ = Pₜ/Pₜ₋₁ − 1`.

- **Daily return:** simple price change between adjacent valid observations. Missing observations are not forward-filled, so returns are not bridged across a missing price.
- **Annualized arithmetic return:** mean daily return multiplied by 252. The 252 figure is the conventional approximate number of trading days in a year.
- **Annualized geometric historical return:** `exp(sum(log(1+r)) × 252/N) − 1` for `N` valid daily observations. Short samples can produce unstable annualized estimates.
- **Daily variance:** sample variance of daily returns (`ddof=1`). Annualized variance is daily variance multiplied by 252.
- **Annualized volatility:** daily sample standard deviation multiplied by `√252`.
- **Covariance/correlation:** estimated from returns aligned by timestamp. Portfolio covariance uses complete overlapping daily-return rows and is scaled by 252; the correlation matrix uses pairwise available observations. Correlation describes historical linear association and does not imply causation.
- **Portfolio expected return:** `wᵀμ`, with arithmetic annualized asset-return estimates `μ`.
- **Portfolio variance:** `wᵀΣw`, with annualized covariance `Σ`; portfolio volatility is its square root.
- **Sharpe ratio:** `(portfolio annualized return − annual risk-free rate) / annualized volatility`. Enter the risk-free rate as a percentage in the sidebar; the calculation converts it to a decimal.
- **Simulation:** NumPy Dirichlet draws produce fully invested long-only weights. A fixed random seed reproduces the simulation for the same inputs.
- **Optimization:** SciPy SLSQP minimizes portfolio variance or maximizes the Sharpe ratio subject to each weight being from 0 to 1 and weights summing to 1. These are historical estimates under the selected assets and period, not universal optima.

Arithmetic and geometric returns are presented separately. Latest daily return is distinguished from average daily return. Missing and invalid downloaded values remain missing; the app does not make up market prices.

## Testing

Tests use Python's standard `unittest` framework and do not download market data. Deterministic synthetic observations appear only inside test files and are labeled as test fixtures.

```bat
python -m unittest discover -s tests -v
```

The suite covers account creation, duplicate registration, field validation, password hashing and verification, login rejection, session logout/protection, unauthenticated dashboard entry, SQLite persistence, Yahoo Finance response-shape handling and failure reporting, price and return validation, statistical metrics, covariance/correlation, reproducible portfolio simulation, portfolio weights and metrics, optimizer behavior, and CSV generation.

## Troubleshooting

- **`py -3.12` is not found:** install Python 3.12 and check `py --list`; alternatively point directly to the Python 3.12 executable.
- **Dependency installation fails:** check the network/proxy settings, activate `.venv`, then retry `python -m pip install -r requirements.txt`.
- **No market prices appear:** confirm the network connection, ticker selection, and date range. Yahoo Finance may be unavailable, delayed, or return incomplete data. The dashboard reports a readable error and does not substitute sample data.
- **There are not enough overlapping observations:** expand the period or reduce the selected stocks. Portfolio covariance and optimization require at least two complete overlapping return observations.
- **Optimizer reports unavailable:** use a longer period or a different asset selection; optimization can be undefined for zero volatility or fail to converge for numerically difficult historical estimates.
- **Port is already in use:** run `python -m streamlit run app.py --server.port 8502`.

## Data availability and limitations

Yahoo Finance data is retrieved on demand and may be delayed, incomplete, revised, or unavailable. Adjusted Close is used when supplied; if it is absent, the app clearly identifies use of Close. The app does not model all corporate-action, transaction-cost, tax, liquidity, or market-impact details. Estimates depend on the user-selected period and on overlapping data availability.

## Security considerations

This educational implementation uses PBKDF2-HMAC-SHA256 with a random salt and a 600,000-iteration default. Passwords are not stored in plaintext or placed in session state. SQL statements are parameterized and normalized email is unique. Authentication identity is held in Streamlit session state, which is suitable for a local learning application but is not enterprise identity management and does not provide multi-factor authentication. SQLite is local and intended for this educational application; production deployment requires additional identity, session, database, operational, and deployment security controls. Do not expose the local app or database as a production service.

## Disclaimer

> This application is an educational Statistical Machine Learning project intended for academic analysis and demonstration. It is not investment advice, does not guarantee returns, and should not be used as the sole basis for financial decisions.
