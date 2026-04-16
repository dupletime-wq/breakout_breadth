from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from breakout_breadth import analyzer, config, data_provider
from breakout_breadth.analyzer import BreakoutParams, UniverseBreadthAnalysis


def configure_page() -> None:
    st.set_page_config(page_title=config.APP_TITLE, layout="wide")


def apply_style() -> None:
    st.markdown(
        """
        <style>
        .hero {
            padding: 1.4rem 1.6rem;
            border-radius: 18px;
            background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 55%, #dbeafe 100%);
            color: #f8fafc;
            margin-bottom: 1rem;
        }
        .hero h1 {
            margin: 0;
            font-size: 2rem;
            line-height: 1.15;
        }
        .hero p {
            margin-top: 0.55rem;
            margin-bottom: 0;
            color: rgba(248, 250, 252, 0.88);
        }
        .metric-card {
            padding: 1rem 1.05rem;
            border-radius: 16px;
            background: #f8fafc;
            border: 1px solid rgba(148, 163, 184, 0.28);
            min-height: 110px;
        }
        .metric-card .eyebrow {
            font-size: 0.82rem;
            color: #475569;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .metric-card .value {
            font-size: 1.8rem;
            font-weight: 700;
            margin-top: 0.2rem;
            color: #0f172a;
        }
        .metric-card .caption {
            margin-top: 0.25rem;
            color: #475569;
            font-size: 0.92rem;
        }
        .section-label {
            margin-top: 0.35rem;
            color: #0f172a;
            font-weight: 700;
            font-size: 1.1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def provider_bounds(
    *,
    use_backtest_window: bool,
    backtest_start: date | None,
    backtest_end: date | None,
    hold_days: int,
) -> tuple[date | None, date | None, str | None]:
    if use_backtest_window and backtest_start is not None and backtest_end is not None:
        return (
            backtest_start - timedelta(days=config.LOOKBACK_BUFFER_DAYS),
            backtest_end + timedelta(days=max(config.FORWARD_BUFFER_DAYS, hold_days * 3)),
            None,
        )
    return None, None, config.DEFAULT_HISTORY_PERIOD


def build_controls() -> tuple[BreakoutParams, bool, date | None, date | None, bool]:
    st.sidebar.subheader("Breakout Breadth Controls")
    scan_window = st.sidebar.slider("Scan window (trading days)", min_value=10, max_value=120, value=config.DEFAULT_SCAN_WINDOW, step=5)
    strict_setup = st.sidebar.checkbox("Strict breakout setup", value=False)
    volume_multiplier = st.sidebar.slider("Volume multiplier", min_value=1.0, max_value=3.0, value=config.DEFAULT_VOLUME_MULTIPLIER, step=0.1)
    hold_days = st.sidebar.slider("Hold days", min_value=3, max_value=10, value=config.DEFAULT_HOLD_DAYS, step=1)
    stop_loss_pct = st.sidebar.slider("Stop-loss (%)", min_value=1.0, max_value=10.0, value=config.DEFAULT_STOP_LOSS_PCT * 100.0, step=0.5)
    use_backtest_window = st.sidebar.checkbox("Use custom backtest window", value=False)

    backtest_start = None
    backtest_end = None
    if use_backtest_window:
        default_end = date.today()
        default_start = default_end - timedelta(days=90)
        backtest_start = st.sidebar.date_input("Backtest start", value=default_start)
        backtest_end = st.sidebar.date_input("Backtest end", value=default_end)

    refresh = st.sidebar.button("Refresh cached data", use_container_width=True)
    st.sidebar.caption("Data source: Wikipedia constituents + Yahoo Finance OHLCV. FMP adapter is implemented but disabled by default.")
    st.sidebar.caption("Pending events stay outside the breadth numerator to avoid look-ahead bias.")

    params = BreakoutParams(
        scan_window=int(scan_window),
        volume_multiplier=float(volume_multiplier),
        hold_days=int(hold_days),
        stop_loss_pct=float(stop_loss_pct / 100.0),
        strict_setup=bool(strict_setup),
    )
    return params, bool(use_backtest_window), backtest_start, backtest_end, refresh


@st.cache_data(ttl=config.CACHE_TTL_SECONDS, show_spinner=False)
def load_dashboard_data(
    *,
    params: BreakoutParams,
    use_backtest_window: bool,
    backtest_start: date | None,
    backtest_end: date | None,
    provider_name: str,
    force_refresh: bool = False,
) -> dict[str, Any]:
    constituents = data_provider.load_sp500_constituents(force_refresh=force_refresh)
    fetch_start, fetch_end, period = provider_bounds(
        use_backtest_window=use_backtest_window,
        backtest_start=backtest_start,
        backtest_end=backtest_end,
        hold_days=params.hold_days,
    )
    price_frames, diagnostics = data_provider.download_ohlcv_universe(
        constituents["YahooSymbol"].tolist(),
        provider_name=provider_name,
        start_date=fetch_start,
        end_date=fetch_end,
        period=period,
        force_refresh=force_refresh,
    )
    analysis = analyzer.analyze_universe_breakouts(
        price_frames,
        params,
        start_date=pd.Timestamp(backtest_start) if use_backtest_window and backtest_start is not None else None,
        end_date=pd.Timestamp(backtest_end) if use_backtest_window and backtest_end is not None else None,
    )
    diagnostics_frame = pd.DataFrame(
        [
            {
                "Provider": diagnostics.provider_name,
                "Requested": diagnostics.requested_tickers,
                "Returned": diagnostics.returned_tickers,
                "Missing": len(diagnostics.missing_tickers),
                "Batches": diagnostics.batch_count,
                "Cache hits": diagnostics.cache_hits,
                "Start": diagnostics.start_date or "period",
                "End": diagnostics.end_date or diagnostics.period or "n/a",
            }
        ]
    )
    return {
        "analysis": analysis,
        "price_frames": price_frames,
        "constituents": constituents,
        "provider_diagnostics": diagnostics,
        "provider_diagnostics_frame": diagnostics_frame,
        "provider_name": provider_name,
    }


def render_metric_card(title: str, value: str, caption: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="eyebrow">{title}</div>
            <div class="value">{value}</div>
            <div class="caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_outcome_donut(snapshot: analyzer.BreadthSnapshot) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Success", "Failure", "Pending"],
                values=[snapshot.successes, snapshot.failures, snapshot.pending],
                hole=0.58,
                marker={"colors": ["#16a34a", "#dc2626", "#64748b"]},
                sort=False,
            )
        ]
    )
    fig.update_layout(
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        title="Breakout Outcome Mix",
        legend={"orientation": "h", "y": -0.12},
    )
    return fig


def build_breadth_figure(history: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history["as_of"],
            y=history["breadth_index"],
            mode="lines+markers",
            name="Breadth Index",
            line={"color": "#1d4ed8", "width": 3},
        )
    )
    fig.add_hline(y=config.OVERBOUGHT_BREADTH_THRESHOLD, line_dash="dot", line_color="#0f766e")
    fig.add_hline(y=config.HEALTHY_BREADTH_THRESHOLD, line_dash="dot", line_color="#2563eb")
    fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
    fig.update_layout(
        title="Breadth Index Time Series",
        xaxis_title="As of",
        yaxis_title="Breadth Index",
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
    )
    return fig


def build_daily_counts_figure(daily_counts: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    colors = {"success": "#16a34a", "failure": "#dc2626", "pending": "#64748b"}
    for column in ["success", "failure", "pending"]:
        fig.add_trace(
            go.Bar(
                x=daily_counts["breakout_date"],
                y=daily_counts[column],
                name=column.title(),
                marker_color=colors[column],
            )
        )
    fig.update_layout(
        barmode="stack",
        title="Daily Breakout Counts",
        xaxis_title="Breakout Date",
        yaxis_title="Event Count",
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
    )
    return fig


def _format_pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value * 100.0:.1f}%"


def _format_float(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:,.2f}"


def render_header(snapshot: analyzer.BreadthSnapshot, provider_name: str, params: BreakoutParams, analysis: UniverseBreadthAnalysis) -> None:
    subtitle = (
        f"Provider: {provider_name} | Window: {analysis.diagnostics['window_start'].date()} to "
        f"{analysis.diagnostics['window_end'].date()} | Strict setup: {'On' if params.strict_setup else 'Off'}"
    )
    st.markdown(
        f"""
        <div class="hero">
            <h1>{config.APP_TITLE}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpis(snapshot: analyzer.BreadthSnapshot) -> None:
    attempts = snapshot.attempts
    success_rate = (snapshot.successes / attempts) if attempts else 0.0
    failure_rate = (snapshot.failures / attempts) if attempts else 0.0
    confidence = "Low confidence" if snapshot.confidence_flag else "Normal confidence"
    cards = st.columns(4)
    with cards[0]:
        render_metric_card("Breadth Index", f"{snapshot.breadth_index:.1f}", snapshot.regime)
    with cards[1]:
        render_metric_card("Resolved Attempts", str(snapshot.attempts), f"Pending {snapshot.pending}")
    with cards[2]:
        render_metric_card("Success Rate", f"{success_rate * 100.0:.1f}%", confidence)
    with cards[3]:
        render_metric_card("Failure Rate", f"{failure_rate * 100.0:.1f}%", f"State: {snapshot.regime}")


def render_watchlist(watchlist: pd.DataFrame) -> None:
    st.markdown('<div class="section-label">Failure Watchlist</div>', unsafe_allow_html=True)
    if watchlist.empty:
        st.info("No failed breakouts were detected in the current analysis window.")
        return
    display = watchlist.loc[
        :,
        ["ticker", "breakout_date", "resolved_at", "failure_label", "return_5d", "volume_ratio", "breakout_price"],
    ].copy()
    display.columns = ["Ticker", "Breakout Date", "Failure Date", "Failure Reason", "5D Return", "Volume Ratio", "Breakout Price"]
    display["5D Return"] = display["5D Return"].map(_format_pct)
    display["Volume Ratio"] = display["Volume Ratio"].map(lambda value: f"{value:.2f}x")
    display["Breakout Price"] = display["Breakout Price"].map(_format_float)
    st.dataframe(display, use_container_width=True, hide_index=True)


def render_drilldown(last_events: pd.DataFrame, price_frames: dict[str, pd.DataFrame]) -> None:
    st.markdown('<div class="section-label">Ticker Drill-down</div>', unsafe_allow_html=True)
    if last_events.empty:
        st.info("No breakout events are available for drill-down.")
        return

    selected_ticker = st.selectbox("Drill-down ticker", options=last_events["ticker"].tolist(), index=0)
    selected = last_events.loc[last_events["ticker"] == selected_ticker].iloc[0]
    with st.expander(f"Last event details: {selected_ticker}", expanded=True):
        detail = pd.DataFrame(
            [
                {"Field": "Outcome", "Value": selected["outcome"].title()},
                {"Field": "Breakout date", "Value": pd.Timestamp(selected["breakout_date"]).strftime("%Y-%m-%d")},
                {"Field": "Resolved at", "Value": pd.Timestamp(selected["resolved_at"]).strftime("%Y-%m-%d") if pd.notna(selected["resolved_at"]) else "Pending"},
                {"Field": "Breakout price", "Value": _format_float(selected["breakout_price"])},
                {"Field": "Breakout close", "Value": _format_float(selected["breakout_close"])},
                {"Field": "Volume ratio", "Value": f"{selected['volume_ratio']:.2f}x"},
                {"Field": "Stop-loss", "Value": _format_float(selected["stop_loss_price"])},
                {"Field": "Failure reason", "Value": analyzer.failure_reason_label(selected["failure_reason"])},
                {"Field": "ATR 5%", "Value": _format_pct(selected["atr_pct_5"])},
                {"Field": "ATR 20%", "Value": _format_pct(selected["atr_pct_20"])},
                {"Field": "Close range 5", "Value": _format_float(selected["close_range_5"])},
                {"Field": "Close range 20", "Value": _format_float(selected["close_range_20"])},
                {"Field": "5D return", "Value": _format_pct(selected["return_5d"])},
            ]
        )
        st.dataframe(detail, use_container_width=True, hide_index=True)
        price_frame = price_frames.get(selected_ticker, pd.DataFrame())
        if not price_frame.empty:
            breakout_date = pd.Timestamp(selected["breakout_date"])
            preview = price_frame.loc[
                (price_frame.index >= breakout_date - pd.Timedelta(days=10))
                & (price_frame.index <= breakout_date + pd.Timedelta(days=14))
            ].copy()
            preview = preview.reset_index().rename(columns={"index": "Date"})
            st.dataframe(preview, use_container_width=True, hide_index=True)


def render_diagnostics(result: dict[str, Any], analysis: UniverseBreadthAnalysis) -> None:
    with st.expander("Data diagnostics", expanded=False):
        st.dataframe(result["provider_diagnostics_frame"], use_container_width=True, hide_index=True)
        missing = list(result["provider_diagnostics"].missing_tickers[:20])
        if missing:
            st.warning(f"Missing OHLCV for {len(result['provider_diagnostics'].missing_tickers)} symbols. Sample: {', '.join(missing)}")
        st.caption("Backtest uses current S&P 500 constituents and therefore carries survivorship bias.")
        st.json(
            {
                "tickers_requested": analysis.diagnostics["tickers_requested"],
                "tickers_analyzed": analysis.diagnostics["tickers_analyzed"],
                "tickers_with_events": analysis.diagnostics["tickers_with_events"],
                "event_count": analysis.diagnostics["event_count"],
            }
        )


def main() -> None:
    configure_page()
    apply_style()
    params, use_backtest_window, backtest_start, backtest_end, refresh = build_controls()

    if use_backtest_window and backtest_start is not None and backtest_end is not None and backtest_start > backtest_end:
        st.error("Backtest start must be on or before backtest end.")
        st.stop()

    if refresh:
        st.cache_data.clear()
        data_provider.clear_all_caches()
        st.sidebar.success("Caches cleared. Rerun to fetch fresh data.")

    provider_name = data_provider.select_default_provider_name()
    try:
        with st.spinner("Loading S&P 500 constituents, OHLCV, and breadth analytics..."):
            result = load_dashboard_data(
                params=params,
                use_backtest_window=use_backtest_window,
                backtest_start=backtest_start,
                backtest_end=backtest_end,
                provider_name=provider_name,
                force_refresh=False,
            )
    except Exception as exc:
        st.error("Breakout breadth data could not be loaded.")
        st.info(str(exc))
        st.stop()

    analysis: UniverseBreadthAnalysis = result["analysis"]
    render_header(analysis.snapshot, provider_name, params, analysis)
    render_kpis(analysis.snapshot)

    if analysis.snapshot.attempts == 0 and analysis.snapshot.pending == 0:
        st.warning("No resolved breakouts were found in the current analysis window. Adjust the window or relax the setup.")

    top_col, right_col = st.columns([1.0, 1.35])
    with top_col:
        st.plotly_chart(build_outcome_donut(analysis.snapshot), use_container_width=True, config={"displaylogo": False})
    with right_col:
        st.plotly_chart(build_breadth_figure(analysis.history), use_container_width=True, config={"displaylogo": False})

    st.plotly_chart(build_daily_counts_figure(analysis.daily_counts), use_container_width=True, config={"displaylogo": False})
    render_watchlist(analysis.watchlist)
    render_drilldown(analysis.last_events, result["price_frames"])
    render_diagnostics(result, analysis)


if __name__ == "__main__":
    main()
