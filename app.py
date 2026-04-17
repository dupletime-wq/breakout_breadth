from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import sys
from typing import Any, Callable

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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


def trading_days_to_calendar_days(trading_days: int) -> int:
    return max(config.LOOKBACK_BUFFER_DAYS, trading_days * config.TRADING_TO_CALENDAR_RATIO)


def required_prefetch_trading_days(params: BreakoutParams, *, use_backtest_window: bool) -> int:
    signal_warmup = max(params.lookback_high, params.vol_window, 20) + params.hold_days
    if use_backtest_window:
        return max(params.scan_window + signal_warmup, max(config.RS_LOOKBACK_PERIODS))
    return max((params.scan_window * 2) + signal_warmup, max(config.RS_LOOKBACK_PERIODS))


def provider_bounds(
    *,
    params: BreakoutParams,
    use_backtest_window: bool,
    backtest_start: date | None,
    backtest_end: date | None,
) -> tuple[date | None, date | None, str | None, int]:
    prefetch_trading_days = required_prefetch_trading_days(params, use_backtest_window=use_backtest_window)
    prefetch_calendar_days = trading_days_to_calendar_days(prefetch_trading_days)
    forward_days = max(config.FORWARD_BUFFER_DAYS, params.hold_days * 3)

    if use_backtest_window and backtest_start is not None and backtest_end is not None:
        return (
            backtest_start - timedelta(days=prefetch_calendar_days),
            backtest_end + timedelta(days=forward_days),
            None,
            prefetch_trading_days,
        )

    today = date.today()
    return (
        today - timedelta(days=prefetch_calendar_days),
        today,
        None,
        prefetch_trading_days,
    )


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


def _dashboard_request_key(
    *,
    params: BreakoutParams,
    use_backtest_window: bool,
    backtest_start: date | None,
    backtest_end: date | None,
    provider_name: str,
) -> tuple[Any, ...]:
    return (
        params.lookback_high,
        params.scan_window,
        params.vol_window,
        params.volume_multiplier,
        params.hold_days,
        params.stop_loss_pct,
        params.strict_setup,
        use_backtest_window,
        backtest_start.isoformat() if backtest_start is not None else None,
        backtest_end.isoformat() if backtest_end is not None else None,
        provider_name,
    )


def load_dashboard_data(
    *,
    params: BreakoutParams,
    use_backtest_window: bool,
    backtest_start: date | None,
    backtest_end: date | None,
    provider_name: str,
    force_refresh: bool = False,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    if progress_callback is not None:
        progress_callback(0.05, "Loading S&P 500 constituents...")
    constituents = data_provider.load_sp500_constituents(force_refresh=force_refresh)

    fetch_start, fetch_end, period, prefetch_trading_days = provider_bounds(
        params=params,
        use_backtest_window=use_backtest_window,
        backtest_start=backtest_start,
        backtest_end=backtest_end,
    )

    if progress_callback is not None:
        progress_callback(0.12, "Preparing OHLCV universe request...")

    def update_universe_progress(batch_index: int, batch_count: int, source: str) -> None:
        if progress_callback is None:
            return
        source_label = "Reading cached OHLCV" if source == "cache" else "Downloading OHLCV"
        progress = 0.12 + (0.56 * (batch_index / max(batch_count, 1)))
        progress_callback(progress, f"{source_label}: batch {batch_index}/{batch_count}")

    price_frames, diagnostics = data_provider.download_ohlcv_universe(
        constituents["YahooSymbol"].tolist(),
        provider_name=provider_name,
        start_date=fetch_start,
        end_date=fetch_end,
        period=period,
        force_refresh=force_refresh,
        progress_callback=update_universe_progress,
    )

    if progress_callback is not None:
        progress_callback(0.74, f"Loading benchmark history for {config.BENCHMARK_SYMBOL}...")
    benchmark_frames, benchmark_diagnostics = data_provider.download_ohlcv_universe(
        [config.BENCHMARK_SYMBOL],
        provider_name=provider_name,
        start_date=fetch_start,
        end_date=fetch_end,
        period=period,
        force_refresh=force_refresh,
    )
    benchmark_frame = benchmark_frames.get(config.BENCHMARK_SYMBOL, pd.DataFrame())

    if progress_callback is not None:
        progress_callback(0.84, "Analyzing breadth, RS ranking, and breakout candidates...")
    analysis = analyzer.analyze_universe_breakouts(
        price_frames,
        params,
        start_date=pd.Timestamp(backtest_start) if use_backtest_window and backtest_start is not None else None,
        end_date=pd.Timestamp(backtest_end) if use_backtest_window and backtest_end is not None else None,
        benchmark_frame=benchmark_frame,
    )

    if progress_callback is not None:
        progress_callback(0.95, "Finalizing dashboard tables and charts...")

    diagnostics_frame = pd.DataFrame(
        [
            {
                "Scope": "Universe",
                "Provider": diagnostics.provider_name,
                "Requested": diagnostics.requested_tickers,
                "Returned": diagnostics.returned_tickers,
                "Missing": len(diagnostics.missing_tickers),
                "Batches": diagnostics.batch_count,
                "Cache hits": diagnostics.cache_hits,
                "Start": diagnostics.start_date or "period",
                "End": diagnostics.end_date or diagnostics.period or "n/a",
            },
            {
                "Scope": f"Benchmark ({config.BENCHMARK_SYMBOL})",
                "Provider": benchmark_diagnostics.provider_name,
                "Requested": benchmark_diagnostics.requested_tickers,
                "Returned": benchmark_diagnostics.returned_tickers,
                "Missing": len(benchmark_diagnostics.missing_tickers),
                "Batches": benchmark_diagnostics.batch_count,
                "Cache hits": benchmark_diagnostics.cache_hits,
                "Start": benchmark_diagnostics.start_date or "period",
                "End": benchmark_diagnostics.end_date or benchmark_diagnostics.period or "n/a",
            },
        ]
    )

    if progress_callback is not None:
        progress_callback(1.0, "Dashboard ready.")

    return {
        "analysis": analysis,
        "price_frames": price_frames,
        "constituents": constituents,
        "provider_diagnostics": diagnostics,
        "benchmark_diagnostics": benchmark_diagnostics,
        "provider_diagnostics_frame": diagnostics_frame,
        "provider_name": provider_name,
        "benchmark_frame": benchmark_frame,
        "prefetch_trading_days": prefetch_trading_days,
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


def build_ticker_context_figure(
    ticker: str,
    price_frame: pd.DataFrame,
    params: BreakoutParams,
    *,
    as_of: pd.Timestamp,
    breakout_level: float | None = None,
    breakout_date: pd.Timestamp | None = None,
) -> go.Figure:
    if price_frame.empty:
        return go.Figure()

    clipped = price_frame.loc[price_frame.index <= pd.Timestamp(as_of)].copy()
    if clipped.empty:
        clipped = price_frame.copy()
    clipped.index = pd.to_datetime(clipped.index)
    clipped = clipped.sort_index()

    resistance = clipped["Close"].rolling(params.lookback_high).max().shift(1)
    avg_volume = clipped["Volume"].rolling(params.vol_window).mean().shift(1)
    chart_frame = clipped.tail(config.TICKER_CHART_TRADING_DAYS).copy()
    resistance = resistance.loc[chart_frame.index]
    avg_volume = avg_volume.loc[chart_frame.index]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.72, 0.28],
    )
    fig.add_trace(
        go.Candlestick(
            x=chart_frame.index,
            open=chart_frame["Open"],
            high=chart_frame["High"],
            low=chart_frame["Low"],
            close=chart_frame["Close"],
            name=ticker,
            increasing_line_color="#16a34a",
            decreasing_line_color="#dc2626",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=chart_frame.index,
            y=resistance,
            name=f"{params.lookback_high}D breakout line",
            line={"color": "#2563eb", "width": 2, "dash": "dash"},
        ),
        row=1,
        col=1,
    )
    if breakout_level is not None and not pd.isna(breakout_level):
        fig.add_trace(
            go.Scatter(
                x=[chart_frame.index.min(), chart_frame.index.max()],
                y=[breakout_level, breakout_level],
                name="Selected level",
                line={"color": "#0f766e", "width": 2, "dash": "dot"},
            ),
            row=1,
            col=1,
        )
    if breakout_date is not None and not pd.isna(breakout_date):
        fig.add_vline(x=pd.Timestamp(breakout_date), row=1, col=1, line_dash="dot", line_color="#334155")

    fig.add_trace(
        go.Bar(
            x=chart_frame.index,
            y=chart_frame["Volume"],
            name="Volume",
            marker_color="#94a3b8",
            opacity=0.85,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=chart_frame.index,
            y=avg_volume,
            name=f"{params.vol_window}D avg volume",
            line={"color": "#1d4ed8", "width": 2},
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=f"{ticker} Price And Volume Context",
        margin={"l": 10, "r": 10, "t": 45, "b": 10},
        xaxis_rangeslider_visible=False,
        legend={"orientation": "h", "y": 1.08},
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig


def _format_pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value * 100.0:.1f}%"


def _format_float(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:,.2f}"


def _format_score(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.1f}"


def render_header(snapshot: analyzer.BreadthSnapshot, provider_name: str, params: BreakoutParams, analysis: UniverseBreadthAnalysis, prefetch_trading_days: int) -> None:
    subtitle = (
        f"Provider: {provider_name} | Window: {analysis.diagnostics['window_start'].date()} to "
        f"{analysis.diagnostics['window_end'].date()} | Strict setup: {'On' if params.strict_setup else 'Off'} | "
        f"Warmup loaded: {prefetch_trading_days} trading days"
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


def render_drilldown(last_events: pd.DataFrame, price_frames: dict[str, pd.DataFrame], params: BreakoutParams, as_of: pd.Timestamp) -> None:
    st.markdown('<div class="section-label">Recent Breakout Drill-down</div>', unsafe_allow_html=True)
    if last_events.empty:
        st.info("No breakout events are available for drill-down.")
        return

    selected_ticker = st.selectbox("Recent breakout ticker", options=last_events["ticker"].tolist(), index=0, key="recent_breakout_ticker")
    selected = last_events.loc[last_events["ticker"] == selected_ticker].iloc[0]
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
            {"Field": "5D return", "Value": _format_pct(selected["return_5d"])},
        ]
    )
    st.dataframe(detail, use_container_width=True, hide_index=True)

    price_frame = price_frames.get(selected_ticker, pd.DataFrame())
    if price_frame.empty:
        st.info("Price history is unavailable for the selected ticker.")
        return
    figure = build_ticker_context_figure(
        selected_ticker,
        price_frame,
        params,
        as_of=as_of,
        breakout_level=selected["breakout_price"],
        breakout_date=pd.Timestamp(selected["breakout_date"]),
    )
    st.plotly_chart(figure, use_container_width=True, config={"displaylogo": False})


def render_rs_ranking(rs_ranking: pd.DataFrame, price_frames: dict[str, pd.DataFrame], params: BreakoutParams, as_of: pd.Timestamp) -> None:
    st.markdown('<div class="section-label">RS Ranking</div>', unsafe_allow_html=True)
    st.caption(f"Relative strength is measured versus {config.BENCHMARK_SYMBOL} using weighted 21D/63D/126D/252D relative returns.")
    if rs_ranking.empty:
        st.info("RS ranking could not be calculated with the available history.")
        return

    ranking = rs_ranking.head(config.RS_TOP_N).copy()
    display = ranking.loc[:, ["rs_rank", "ticker", "rs_percentile", "relative_21d", "relative_63d", "relative_126d", "relative_252d", "price"]].copy()
    display.columns = ["Rank", "Ticker", "RS Percentile", "Rel 21D", "Rel 63D", "Rel 126D", "Rel 252D", "Price"]
    for column in ["Rel 21D", "Rel 63D", "Rel 126D", "Rel 252D"]:
        display[column] = display[column].map(_format_pct)
    display["RS Percentile"] = display["RS Percentile"].map(lambda value: f"{value:.1f}")
    display["Price"] = display["Price"].map(_format_float)
    st.dataframe(display, use_container_width=True, hide_index=True)

    selected_ticker = st.selectbox("RS ticker chart", options=ranking["ticker"].tolist(), index=0, key="rs_ticker")
    selected = ranking.loc[ranking["ticker"] == selected_ticker].iloc[0]
    detail = pd.DataFrame(
        [
            {"Field": "RS rank", "Value": int(selected["rs_rank"])},
            {"Field": "RS percentile", "Value": f"{selected['rs_percentile']:.1f}"},
            {"Field": "RS score", "Value": f"{selected['rs_score'] * 100.0:.2f}"},
            {"Field": "Rel 21D", "Value": _format_pct(selected["relative_21d"])},
            {"Field": "Rel 63D", "Value": _format_pct(selected["relative_63d"])},
            {"Field": "Rel 126D", "Value": _format_pct(selected["relative_126d"])},
            {"Field": "Rel 252D", "Value": _format_pct(selected["relative_252d"])},
        ]
    )
    st.dataframe(detail, use_container_width=True, hide_index=True)

    price_frame = price_frames.get(selected_ticker, pd.DataFrame())
    if price_frame.empty:
        st.info("Price history is unavailable for the selected ticker.")
        return
    st.plotly_chart(
        build_ticker_context_figure(selected_ticker, price_frame, params, as_of=as_of),
        use_container_width=True,
        config={"displaylogo": False},
    )


def render_waiting_watchlist(waiting_watchlist: pd.DataFrame, price_frames: dict[str, pd.DataFrame], params: BreakoutParams, as_of: pd.Timestamp) -> None:
    st.markdown('<div class="section-label">Breakout Waiting Watchlist</div>', unsafe_allow_html=True)
    st.caption("Candidates are near their breakout level, have strong RS, and are scored for contraction and proximity.")
    if waiting_watchlist.empty:
        st.info("No high-RS breakout candidates are currently near their trigger level.")
        return

    candidates = waiting_watchlist.head(config.WAITING_WATCHLIST_TOP_N).copy()
    display = candidates.loc[:, ["ticker", "waiting_score", "rs_percentile", "distance_to_breakout_pct", "volume_ratio", "breakout_level", "strict_ready"]].copy()
    display.columns = ["Ticker", "Waiting Score", "RS Percentile", "Distance To Breakout", "Volume Ratio", "Breakout Level", "Strict Ready"]
    display["Waiting Score"] = display["Waiting Score"].map(_format_score)
    display["RS Percentile"] = display["RS Percentile"].map(lambda value: f"{value:.1f}")
    display["Distance To Breakout"] = display["Distance To Breakout"].map(_format_pct)
    display["Volume Ratio"] = display["Volume Ratio"].map(lambda value: f"{value:.2f}x")
    display["Breakout Level"] = display["Breakout Level"].map(_format_float)
    display["Strict Ready"] = display["Strict Ready"].map(lambda value: "Yes" if bool(value) else "No")
    st.dataframe(display, use_container_width=True, hide_index=True)

    selected_ticker = st.selectbox("Waiting watchlist ticker chart", options=candidates["ticker"].tolist(), index=0, key="waiting_ticker")
    selected = candidates.loc[candidates["ticker"] == selected_ticker].iloc[0]
    detail = pd.DataFrame(
        [
            {"Field": "Waiting score", "Value": _format_score(selected["waiting_score"])},
            {"Field": "RS percentile", "Value": f"{selected['rs_percentile']:.1f}"},
            {"Field": "Distance to breakout", "Value": _format_pct(selected["distance_to_breakout_pct"])},
            {"Field": "Breakout level", "Value": _format_float(selected["breakout_level"])},
            {"Field": "Current close", "Value": _format_float(selected["current_close"])},
            {"Field": "Volume ratio", "Value": f"{selected['volume_ratio']:.2f}x"},
            {"Field": "Strict ready", "Value": "Yes" if bool(selected["strict_ready"]) else "No"},
        ]
    )
    st.dataframe(detail, use_container_width=True, hide_index=True)

    price_frame = price_frames.get(selected_ticker, pd.DataFrame())
    if price_frame.empty:
        st.info("Price history is unavailable for the selected ticker.")
        return
    st.plotly_chart(
        build_ticker_context_figure(
            selected_ticker,
            price_frame,
            params,
            as_of=as_of,
            breakout_level=selected["breakout_level"],
        ),
        use_container_width=True,
        config={"displaylogo": False},
    )


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
                "benchmark_symbol": analysis.diagnostics.get("benchmark_symbol"),
                "benchmark_available": analysis.diagnostics.get("benchmark_available"),
                "warmup_trading_days": result["prefetch_trading_days"],
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
        st.session_state.pop("dashboard_result", None)
        st.session_state.pop("dashboard_request_key", None)
        st.sidebar.success("Caches cleared. Fresh data will be fetched for this run.")

    provider_name = data_provider.select_default_provider_name()
    request_key = _dashboard_request_key(
        params=params,
        use_backtest_window=use_backtest_window,
        backtest_start=backtest_start,
        backtest_end=backtest_end,
        provider_name=provider_name,
    )

    result = None
    if st.session_state.get("dashboard_request_key") == request_key:
        result = st.session_state.get("dashboard_result")

    if result is None:
        progress_text = st.empty()
        progress_bar = st.progress(0)

        def update_progress(progress: float, message: str) -> None:
            progress_bar.progress(min(100, max(0, int(progress * 100))))
            progress_text.caption(message)

        try:
            result = load_dashboard_data(
                params=params,
                use_backtest_window=use_backtest_window,
                backtest_start=backtest_start,
                backtest_end=backtest_end,
                provider_name=provider_name,
                force_refresh=False,
                progress_callback=update_progress,
            )
        except Exception as exc:
            progress_bar.empty()
            progress_text.empty()
            st.error("Breakout breadth data could not be loaded.")
            st.info(str(exc))
            st.stop()

        progress_bar.empty()
        progress_text.empty()
        st.session_state["dashboard_request_key"] = request_key
        st.session_state["dashboard_result"] = result

    analysis: UniverseBreadthAnalysis = result["analysis"]
    render_header(analysis.snapshot, provider_name, params, analysis, result["prefetch_trading_days"])
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

    left_col, right_col = st.columns(2)
    with left_col:
        render_rs_ranking(analysis.rs_ranking, result["price_frames"], params, analysis.snapshot.as_of)
    with right_col:
        render_waiting_watchlist(analysis.waiting_watchlist, result["price_frames"], params, analysis.snapshot.as_of)

    render_drilldown(analysis.last_events, result["price_frames"], params, analysis.snapshot.as_of)
    render_diagnostics(result, analysis)


if __name__ == "__main__":
    main()
