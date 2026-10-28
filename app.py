"""PortfolioIQ: educational Indian equity risk and return dashboard."""

from __future__ import annotations

from datetime import date, timedelta
import logging
from pathlib import Path

import pandas as pd
import streamlit as st

from analytics.market_data import MarketDataError, download_prices
from analytics.portfolio import OptimizationError, maximum_sharpe_portfolio, minimum_volatility_portfolio, simulate_portfolios
from analytics.returns import calculate_daily_returns, correlation_matrix, covariance_matrix, performance_metrics
from analytics.validation import ValidationError, validate_dates, validate_risk_free_rate, validate_simulation_count, validate_tickers
from authentication.auth import AuthService, AuthenticationError, RegistrationError, clear_auth_session, is_authenticated, set_auth_session
from components.ui import allocation_chart, apply_theme, correlation_heatmap, csv_bytes, format_percent, metric_card, price_chart, risk_return_chart
from database.db import Database

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger("portfolioiq")
ROOT = Path(__file__).resolve().parent
DB = Database(ROOT / "data" / "portfolioiq.sqlite3")
DB.initialize()
AUTH = AuthService(DB)

STOCKS = {
    "Reliance Industries · RELIANCE.NS": "RELIANCE.NS",
    "Tata Consultancy Services · TCS.NS": "TCS.NS",
    "Infosys · INFY.NS": "INFY.NS",
    "HDFC Bank · HDFCBANK.NS": "HDFCBANK.NS",
}

st.set_page_config(page_title="PortfolioIQ | Portfolio Analytics", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
apply_theme()


@st.cache_data(ttl=3600, show_spinner=False)
def cached_download(tickers: tuple[str, ...], start_iso: str, end_iso: str) -> tuple[pd.DataFrame, str]:
    return download_prices(list(tickers), date.fromisoformat(start_iso), date.fromisoformat(end_iso))


def dashboard_access_allowed(session_state) -> bool:
    """Testable guard shared by all dashboard entry points."""
    return is_authenticated(session_state)


def show_brand() -> None:
    st.markdown('<div class="brand-kicker">PortfolioIQ · Statistical Machine Learning</div>', unsafe_allow_html=True)


def landing_page() -> None:
    show_brand()
    st.markdown('<div class="hero-title">See risk clearly.<br><span style="color:#d8b46a">Invest with context.</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Portfolio Risk and Return Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="gold-rule"></div>', unsafe_allow_html=True)
    left, right = st.columns([1.2, 1])
    with left:
        st.write("Explore historical Indian equity performance, compare risk, and examine long-only portfolio allocations through clear statistical tools.")
        st.markdown("**Historical data · Risk measurement · Portfolio simulation · Constrained optimization**")
        st.caption("Educational analysis using Yahoo Finance data. Results depend on the selected period and are not forecasts.")
    with right:
        st.markdown("### Your workspace")
        st.write("Create a local account to access the interactive dashboard.")
        a, b = st.columns(2)
        if a.button("Log in", type="primary", use_container_width=True):
            st.session_state["page"] = "login"
            st.rerun()
        if b.button("Create account", use_container_width=True):
            st.session_state["page"] = "signup"
            st.rerun()


def login_page() -> None:
    show_brand()
    st.title("Welcome back")
    st.caption("Log in to continue to your private analysis workspace.")
    with st.form("login_form"):
        email = st.text_input("Email address", autocomplete="email")
        password = st.text_input("Password", type="password", autocomplete="current-password")
        submitted = st.form_submit_button("Log in", type="primary", use_container_width=True)
    if submitted:
        try:
            user = AUTH.login(email, password)
            set_auth_session(st.session_state, user)
            st.session_state["page"] = "dashboard"
            st.rerun()
        except AuthenticationError:
            st.error("Invalid email or password.")
        except Exception as exc:
            LOGGER.exception("Login processing failed (%s).", type(exc).__name__)
            st.error("Login is temporarily unavailable. Please try again.")
    if st.button("Create a new account"):
        st.session_state["page"] = "signup"
        st.rerun()
    if st.button("Back to home"):
        st.session_state["page"] = "landing"
        st.rerun()


def signup_page() -> None:
    show_brand()
    st.title("Create your account")
    st.caption("Your password is protected with a salted, slow one-way hash.")
    with st.form("signup_form"):
        full_name = st.text_input("Full name", autocomplete="name")
        email = st.text_input("Email address", autocomplete="email")
        password = st.text_input("Password", type="password", autocomplete="new-password", help="At least 8 characters.")
        confirmation = st.text_input("Confirm password", type="password", autocomplete="new-password")
        submitted = st.form_submit_button("Create account", type="primary", use_container_width=True)
    if submitted:
        try:
            AUTH.register(full_name, email, password, confirmation)
            st.success("Your account is ready. You can now log in.")
            st.session_state["page"] = "login"
        except RegistrationError as exc:
            st.error(str(exc))
        except Exception as exc:
            LOGGER.exception("Registration processing failed (%s).", type(exc).__name__)
            st.error("Account creation is temporarily unavailable. Please try again.")
    if st.button("Already registered? Log in"):
        st.session_state["page"] = "login"
        st.rerun()
    if st.button("Back to home"):
        st.session_state["page"] = "landing"
        st.rerun()


def about_page() -> None:
    st.title("About PortfolioIQ")
    st.write("PortfolioIQ is a college-level Statistical Machine Learning project combining historical price analysis, statistical risk measurement, random portfolio simulation, numerical optimization, and interactive visualization.")
    st.subheader("Methodology")
    st.markdown("1. Retrieve adjusted prices from Yahoo Finance.\n2. Validate observations and calculate simple daily returns.\n3. Estimate annualized returns, sample variance, volatility, covariance, and correlation.\n4. Generate fully invested long-only portfolios with reproducible random weights.\n5. Optimize minimum volatility and, where defined, maximum Sharpe ratio.\n6. Visualize and export computed results.")
    st.subheader("Statistical formulas")
    st.markdown("- Daily simple return: `Pₜ / Pₜ₋₁ − 1`\n- Arithmetic annualized return: `mean(daily returns) × 252`\n- Geometric annualized return: `product(1 + r)^(252/N) − 1`\n- Daily sample variance: sample variance with `ddof=1`; annualized variance: `daily variance × 252`\n- Annualized volatility: `daily standard deviation × √252`\n- Portfolio return: `wᵀμ`; portfolio variance: `wᵀΣw`\n- Sharpe ratio: `(annualized return − annual risk-free rate) / annualized volatility`")
    st.subheader("Assumptions and limitations")
    st.write("The analysis uses about 252 trading days per year, arithmetic daily-return estimates, complete overlapping observations for portfolio risk, long-only fully invested weights, and a user-supplied annual risk-free rate. Historical estimates are uncertain and can change; the model omits some taxes, transaction costs, liquidity limits, and market impact. Correlation describes historical linear co-movement and does not establish causation. Market data may be delayed, incomplete, or unavailable.")
    st.subheader("Security and privacy")
    st.write("This is an educational local SQLite and session-based login implementation. It is not enterprise identity management or multi-factor authentication. A production deployment would need additional controls, secure session handling, deployment hardening, monitoring, and a managed database. Local accounts are stored only on the machine running the app.")
    st.warning("This application is an educational Statistical Machine Learning project intended for academic analysis and demonstration. It is not investment advice, does not guarantee returns, and should not be used as the sole basis for financial decisions.")


def portfolio_weights_frame(weights: pd.Series, metrics: dict[str, float], portfolio_name: str) -> pd.DataFrame:
    frame = weights.rename("Weight").to_frame()
    frame.insert(0, "Portfolio", portfolio_name)
    for key, value in metrics.items():
        frame[key] = value
    return frame


def dashboard() -> None:
    if not dashboard_access_allowed(st.session_state):
        clear_auth_session(st.session_state)
        st.session_state["page"] = "login"
        st.info("Please log in to access the dashboard.")
        login_page()
        return

    with st.sidebar:
        show_brand()
        st.markdown(f"### Hello, {st.session_state.get('full_name', 'Investor')}")
        if st.button("Log out", use_container_width=True):
            clear_auth_session(st.session_state)
            st.session_state["page"] = "landing"
            st.rerun()
        st.divider()
        section = st.radio("Workspace", ["Dashboard", "About"], label_visibility="collapsed")
        if section == "Dashboard":
            st.subheader("Analysis controls")
            selected_labels = st.multiselect("Stocks", list(STOCKS), default=list(STOCKS)[:3])
            today = date.today()
            start_default = today - timedelta(days=365 * 3)
            start_date = st.date_input("Start date", value=start_default, max_value=today - timedelta(days=1))
            end_date = st.date_input("End date", value=today, max_value=today)
            simulation_count = st.number_input("Simulated portfolios", min_value=100, max_value=50_000, value=5_000, step=500)
            random_seed = st.number_input("Random seed", min_value=0, max_value=2**32 - 1, value=42, step=1)
            risk_free_percent = st.number_input("Annual risk-free rate (%)", min_value=-100.0, max_value=100.0, value=6.0, step=0.25, format="%.2f")
            st.caption("252 trading days are used for annualization.")

    if section == "About":
        about_page()
        return

    col_title, col_refresh = st.columns([5, 1])
    with col_title:
        st.markdown('<div class="brand-kicker">PortfolioIQ · Portfolio Analytics</div>', unsafe_allow_html=True)
        st.title("Risk & return overview")
        st.caption(f"Analysis period: {start_date:%d %b %Y} – {end_date:%d %b %Y} · {len(selected_labels)} selected stocks")
    with col_refresh:
        st.write("")
        if st.button("↻ Refresh", use_container_width=True):
            cached_download.clear()
            st.rerun()

    try:
        tickers = validate_tickers([STOCKS[label] for label in selected_labels])
        validate_dates(start_date, end_date)
        validate_simulation_count(int(simulation_count))
        rf_decimal = validate_risk_free_rate(risk_free_percent)
    except (ValidationError, KeyError) as exc:
        st.error(str(exc))
        return
    if (end_date - start_date).days < 4:
        st.error("Choose a date range with at least several calendar days of history.")
        return

    try:
        with st.spinner("Retrieving historical market data…"):
            prices, price_basis = cached_download(tuple(tickers), start_date.isoformat(), end_date.isoformat())
        returns = calculate_daily_returns(prices)
        if returns.dropna(how="all").shape[0] < 2:
            raise ValidationError("The selected period does not contain enough valid daily returns.")
        metrics = performance_metrics(returns)
        correlations = correlation_matrix(returns)
        daily_cov = covariance_matrix(returns, annualized=False)
        annual_cov = covariance_matrix(returns, annualized=True)
    except (MarketDataError, ValidationError) as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        LOGGER.exception("Unexpected analysis failure (%s).", type(exc).__name__)
        st.error("The analysis could not be completed. Please adjust the selection or try again.")
        return

    st.caption(f"Price series: {price_basis}. Yahoo Finance data availability and corporate-action adjustments may vary.")
    st.subheader("Latest observations")
    price_cols = st.columns(len(prices.columns))
    for column, ticker in zip(price_cols, prices.columns):
        latest = prices[ticker].dropna()
        latest_return = returns[ticker].dropna()
        with column:
            metric_card(f"{ticker} · latest price", f"₹{latest.iloc[-1]:,.2f}" if len(latest) else "—")
            st.caption(f"As of {latest.index[-1]:%d %b %Y}" if len(latest) else "No valid price")

    st.subheader("Stock-level performance")
    st.caption("Latest daily return is the final observed one-day return; it is distinct from average daily return.")
    display_metrics = metrics.copy()
    for col in ["Latest Daily Return", "Mean Daily Return", "Annualized Arithmetic Return", "Annualized Geometric Return", "Daily Volatility", "Annualized Volatility"]:
        display_metrics[col] = display_metrics[col].map(format_percent)
    for col in ["Daily Variance", "Annualized Variance"]:
        display_metrics[col] = display_metrics[col].map(lambda value: f"{value:.8f}" if pd.notna(value) else "—")
    st.dataframe(display_metrics, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.pyplot(price_chart(prices, f"Historical {price_basis} prices", "Price (₹)"), use_container_width=True)
    with right:
        st.pyplot(price_chart(returns * 100, "Daily simple returns", "Daily return (%)"), use_container_width=True)
        st.caption("Daily returns measure percentage changes between adjacent valid price observations; prices show levels.")

    st.subheader("Historical correlation")
    st.pyplot(correlation_heatmap(correlations), use_container_width=True)
    st.caption("Pearson correlation measures the strength and direction of the historical linear relationship between daily returns; it does not establish causation.")
    with st.expander("Covariance matrices"):
        cov_left, cov_right = st.columns(2)
        cov_left.markdown("**Daily sample covariance**")
        cov_left.dataframe(daily_cov, use_container_width=True)
        cov_right.markdown("**Annualized covariance (× 252)**")
        cov_right.dataframe(annual_cov, use_container_width=True)

    st.subheader("Portfolio analysis")
    try:
        simulations = simulate_portfolios(returns, int(simulation_count), int(random_seed), rf_decimal)
        min_weights, min_metrics = minimum_volatility_portfolio(returns)
    except (ValidationError, OptimizationError) as exc:
        st.error(f"Portfolio analysis is unavailable: {exc}")
        simulations = None
        min_weights = min_metrics = None
    if simulations is not None:
        max_weights = max_metrics = None
        try:
            max_weights, max_metrics = maximum_sharpe_portfolio(returns, rf_decimal)
        except (ValidationError, OptimizationError) as exc:
            st.warning(f"Maximum-Sharpe optimization is unavailable for these inputs: {exc}")
        st.pyplot(risk_return_chart(simulations, min_metrics, max_metrics), use_container_width=True)
        st.caption("Returns and volatility are annualized decimals in the chart. Optimized portfolios are historical, long-only solutions and are not forecasts.")
        opt_cols = st.columns(2 if max_weights is not None else 1)
        with opt_cols[0]:
            st.markdown("#### Minimum-volatility portfolio")
            st.metric("Expected annualized return", format_percent(min_metrics["Expected Annualized Return"]))
            st.metric("Annualized volatility", format_percent(min_metrics["Annualized Volatility"]))
            st.metric("Annualized variance", f"{min_metrics['Annualized Variance']:.8f}")
            st.dataframe(min_weights.rename("Weight").to_frame().style.format({"Weight": "{:.2%}"}), use_container_width=True)
            st.pyplot(allocation_chart(min_weights, "Minimum-volatility allocation"), use_container_width=True)
        if max_weights is not None:
            with opt_cols[1]:
                st.markdown("#### Maximum-Sharpe portfolio")
                st.metric("Expected annualized return", format_percent(max_metrics["Expected Annualized Return"]))
                st.metric("Annualized volatility", format_percent(max_metrics["Annualized Volatility"]))
                st.metric("Sharpe ratio", f"{max_metrics['Sharpe Ratio']:.3f}")
                st.dataframe(max_weights.rename("Weight").to_frame().style.format({"Weight": "{:.2%}"}), use_container_width=True)
                st.pyplot(allocation_chart(max_weights, "Maximum-Sharpe allocation"), use_container_width=True)
        with st.expander("Random simulation results"):
            st.dataframe(simulations.head(100), use_container_width=True)
            st.caption(f"Showing the first 100 of {len(simulations):,} portfolios. Download includes all rows.")

    st.subheader("Export analysis")
    exports: list[tuple[str, pd.DataFrame, str]] = [
        ("Historical adjusted prices", prices, "portfolioiq_historical_prices.csv"),
        ("Daily returns", returns, "portfolioiq_daily_returns.csv"),
        ("Stock performance metrics", metrics, "portfolioiq_stock_metrics.csv"),
        ("Correlation matrix", correlations, "portfolioiq_correlation_matrix.csv"),
        ("Daily covariance", daily_cov, "portfolioiq_daily_covariance.csv"),
        ("Annualized covariance", annual_cov, "portfolioiq_annualized_covariance.csv"),
    ]
    if simulations is not None:
        exports.append(("Portfolio simulations", simulations, "portfolioiq_portfolio_simulations.csv"))
    if min_weights is not None:
        exports.append(("Minimum-volatility portfolio", portfolio_weights_frame(min_weights, min_metrics, "Minimum volatility"), "portfolioiq_minimum_volatility.csv"))
    if max_weights is not None:
        exports.append(("Maximum-Sharpe portfolio", portfolio_weights_frame(max_weights, max_metrics, "Maximum Sharpe"), "portfolioiq_maximum_sharpe.csv"))
    export_cols = st.columns(min(3, len(exports)))
    for index, (label, frame, filename) in enumerate(exports):
        with export_cols[index % len(export_cols)]:
            st.download_button(label, csv_bytes(frame), file_name=filename, mime="text/csv", key=f"download_{filename}", use_container_width=True)

    st.divider()
    st.warning("This application is an educational Statistical Machine Learning project intended for academic analysis and demonstration. It is not investment advice, does not guarantee returns, and should not be used as the sole basis for financial decisions.")


def main() -> None:
    if "page" not in st.session_state:
        st.session_state["page"] = "dashboard" if is_authenticated(st.session_state) else "landing"
    if is_authenticated(st.session_state):
        # Every authenticated app rerun enters through the guarded dashboard.
        dashboard()
        return
    clear_auth_session(st.session_state)
    page = st.session_state.get("page", "landing")
    if page == "login":
        login_page()
    elif page == "signup":
        signup_page()
    else:
        landing_page()


if __name__ == "__main__":
    main()
