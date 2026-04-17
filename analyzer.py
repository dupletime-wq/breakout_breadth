from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from math import isnan
import os
from typing import Any, Mapping

import numpy as np
import pandas as pd

from . import config


@dataclass(frozen=True)
class BreakoutParams:
    lookback_high: int = config.DEFAULT_LOOKBACK_HIGH
    scan_window: int = config.DEFAULT_SCAN_WINDOW
    vol_window: int = config.DEFAULT_VOLUME_WINDOW
    volume_multiplier: float = config.DEFAULT_VOLUME_MULTIPLIER
    hold_days: int = config.DEFAULT_HOLD_DAYS
    stop_loss_pct: float = config.DEFAULT_STOP_LOSS_PCT
    strict_setup: bool = False


@dataclass(frozen=True)
class BreakoutEvent:
    ticker: str
    breakout_date: pd.Timestamp
    breakout_price: float
    outcome: str
    failure_reason: str | None
    return_5d: float | None
    volume_ratio: float
    breakout_close: float
    stop_loss_price: float
    resolved_at: pd.Timestamp | None
    resolution_days: int | None
    hold_days: int
    strict_setup: bool
    atr_pct_5: float | None
    atr_pct_20: float | None
    close_range_5: float | None
    close_range_20: float | None
    previous_close: float | None


@dataclass(frozen=True)
class BreadthSnapshot:
    as_of: pd.Timestamp
    attempts: int
    successes: int
    failures: int
    pending: int
    breadth_index: float
    regime: str
    confidence_flag: bool


@dataclass(frozen=True)
class UniverseBreadthAnalysis:
    params: BreakoutParams
    snapshot: BreadthSnapshot
    history: pd.DataFrame
    daily_counts: pd.DataFrame
    watchlist: pd.DataFrame
    rs_ranking: pd.DataFrame
    waiting_watchlist: pd.DataFrame
    last_events: pd.DataFrame
    events: tuple[BreakoutEvent, ...]
    diagnostics: dict[str, Any]


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if isnan(numeric):
        return None
    return numeric


def _event_to_row(event: BreakoutEvent) -> dict[str, Any]:
    row = asdict(event)
    row["breakout_date"] = pd.Timestamp(event.breakout_date)
    row["resolved_at"] = pd.Timestamp(event.resolved_at) if event.resolved_at is not None else pd.NaT
    row["failure_label"] = failure_reason_label(event.failure_reason)
    return row


def _infer_regime(attempts: int, breadth_index: float, success_rate: float) -> str:
    if breadth_index >= config.OVERBOUGHT_BREADTH_THRESHOLD and success_rate >= config.OVERBOUGHT_SUCCESS_RATE:
        return "Overbought"
    if breadth_index >= config.HEALTHY_BREADTH_THRESHOLD:
        return "Healthy"
    return "Warning"


def _true_range(frame: pd.DataFrame) -> pd.Series:
    prev_close = frame["Close"].shift(1)
    return pd.concat(
        [
            frame["High"] - frame["Low"],
            (frame["High"] - prev_close).abs(),
            (frame["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def _prepare_frame(price_df: pd.DataFrame) -> pd.DataFrame:
    if price_df.empty:
        return pd.DataFrame(columns=config.REQUIRED_OHLCV_COLUMNS)
    frame = price_df.copy()
    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()
    frame = frame.loc[~frame.index.duplicated(keep="last")]
    missing = [column for column in config.REQUIRED_OHLCV_COLUMNS if column not in frame.columns]
    if missing:
        return pd.DataFrame(columns=config.REQUIRED_OHLCV_COLUMNS)
    return frame.loc[:, list(config.REQUIRED_OHLCV_COLUMNS)].dropna(subset=["Open", "High", "Low", "Close"])


def failure_reason_label(reason: Any) -> str:
    if reason is None:
        return "Resolved"
    if not isinstance(reason, str):
        try:
            if pd.isna(reason):
                return "Resolved"
        except TypeError:
            pass
        reason = str(reason)
    if reason == config.FAILURE_STOP_LOSS:
        return "Stop-loss triggered"
    if reason == config.FAILURE_CLOSE_LOST:
        return "Lost breakout level"
    return reason.replace("_", " ").title()


def _frame_until(frame: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    return frame.loc[frame.index <= pd.Timestamp(as_of)].copy()


def _trailing_return(frame: pd.DataFrame, lookback: int, as_of: pd.Timestamp) -> float | None:
    clipped = _frame_until(frame, as_of)
    if clipped.empty or len(clipped) <= lookback:
        return None
    current_close = clipped["Close"].iloc[-1]
    base_close = clipped["Close"].iloc[-lookback - 1]
    if pd.isna(current_close) or pd.isna(base_close) or float(base_close) == 0.0:
        return None
    return float(current_close / base_close - 1.0)


def _build_relative_strength_table(
    valid_items: list[tuple[str, pd.DataFrame]],
    benchmark_frame: pd.DataFrame | None,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    weights = {21: 0.2, 63: 0.3, 126: 0.3, 252: 0.2}
    benchmark_returns = {
        lookback: _trailing_return(benchmark_frame, lookback, as_of) if benchmark_frame is not None and not benchmark_frame.empty else None
        for lookback in config.RS_LOOKBACK_PERIODS
    }

    rows: list[dict[str, Any]] = []
    for ticker, frame in valid_items:
        clipped = _frame_until(frame, as_of)
        if clipped.empty:
            continue
        weighted_score = 0.0
        weight_total = 0.0
        row: dict[str, Any] = {
            "ticker": ticker,
            "price": float(clipped["Close"].iloc[-1]),
        }
        for lookback in config.RS_LOOKBACK_PERIODS:
            stock_return = _trailing_return(clipped, lookback, as_of)
            benchmark_return = benchmark_returns.get(lookback)
            relative_return = None if stock_return is None else stock_return - benchmark_return if benchmark_return is not None else stock_return
            row[f"return_{lookback}d"] = stock_return
            row[f"relative_{lookback}d"] = relative_return
            if relative_return is not None:
                weighted_score += relative_return * weights.get(lookback, 0.0)
                weight_total += weights.get(lookback, 0.0)
        if weight_total == 0.0:
            continue
        row["rs_score"] = float(weighted_score / weight_total)
        rows.append(row)

    ranking = pd.DataFrame(rows)
    if ranking.empty:
        return ranking

    ranking["rs_percentile"] = ranking["rs_score"].rank(pct=True, method="average") * 100.0
    ranking["rs_rank"] = ranking["rs_score"].rank(method="dense", ascending=False).astype(int)
    ranking = ranking.sort_values(["rs_score", "ticker"], ascending=[False, True]).reset_index(drop=True)
    ordered_columns = [
        "rs_rank",
        "ticker",
        "price",
        "rs_score",
        "rs_percentile",
        "relative_21d",
        "relative_63d",
        "relative_126d",
        "relative_252d",
        "return_21d",
        "return_63d",
        "return_126d",
        "return_252d",
    ]
    return ranking.loc[:, ordered_columns]


def _build_waiting_watchlist(
    valid_items: list[tuple[str, pd.DataFrame]],
    params: BreakoutParams,
    as_of: pd.Timestamp,
    rs_ranking: pd.DataFrame,
) -> pd.DataFrame:
    rs_lookup = rs_ranking.set_index("ticker") if not rs_ranking.empty else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    minimum_rows = max(params.lookback_high, params.vol_window, 20) + 2

    for ticker, frame in valid_items:
        clipped = _frame_until(frame, as_of)
        if clipped.empty or len(clipped) < minimum_rows:
            continue

        resistance = clipped["Close"].rolling(params.lookback_high).max().shift(1).iloc[-1]
        avg_volume = clipped["Volume"].rolling(params.vol_window).mean().shift(1).iloc[-1]
        if pd.isna(resistance) or pd.isna(avg_volume) or float(resistance) <= 0.0 or float(avg_volume) <= 0.0:
            continue

        current_close = float(clipped["Close"].iloc[-1])
        current_high = float(clipped["High"].iloc[-1])
        volume_ratio = float(clipped["Volume"].iloc[-1] / avg_volume) if float(avg_volume) else 0.0
        distance_to_breakout_pct = float((float(resistance) - current_close) / float(resistance))
        if distance_to_breakout_pct < 0.0 or distance_to_breakout_pct > config.WAITING_WATCHLIST_MAX_DISTANCE_PCT:
            continue

        tr = _true_range(clipped)
        atr_5 = tr.rolling(5).mean().shift(1).iloc[-1]
        atr_20 = tr.rolling(20).mean().shift(1).iloc[-1]
        close_range_5 = (clipped["Close"].rolling(5).max() - clipped["Close"].rolling(5).min()).shift(1).iloc[-1]
        close_range_20 = (clipped["Close"].rolling(20).max() - clipped["Close"].rolling(20).min()).shift(1).iloc[-1]
        prior_close = clipped["Close"].shift(1).iloc[-1]

        atr_contraction = 0.0
        if pd.notna(atr_5) and pd.notna(atr_20) and float(atr_20) > 0.0:
            atr_contraction = max(0.0, min(1.0, float((atr_20 - atr_5) / atr_20)))

        range_tightness = 0.0
        if pd.notna(close_range_5) and pd.notna(close_range_20) and float(close_range_20) > 0.0:
            range_tightness = max(0.0, min(1.0, float(1.0 - (close_range_5 / close_range_20))))

        rs_percentile = float(rs_lookup.at[ticker, "rs_percentile"]) if not rs_lookup.empty and ticker in rs_lookup.index else 0.0
        if rs_percentile < config.WAITING_WATCHLIST_MIN_RS_PERCENTILE:
            continue

        proximity_score = max(0.0, 1.0 - (distance_to_breakout_pct / config.WAITING_WATCHLIST_MAX_DISTANCE_PCT))
        volume_score = max(0.0, min(1.0, volume_ratio / max(params.volume_multiplier, 1.0)))
        rs_score_component = max(0.0, min(1.0, rs_percentile / 100.0))
        strict_ready = bool(
            pd.notna(atr_5)
            and pd.notna(atr_20)
            and pd.notna(close_range_5)
            and pd.notna(close_range_20)
            and pd.notna(prior_close)
            and float(atr_5) < float(atr_20)
            and float(close_range_5) <= float(close_range_20) * 0.60
            and float(prior_close) >= float(resistance) * 0.98
        )
        waiting_score = float(
            100.0
            * (
                0.45 * proximity_score
                + 0.25 * rs_score_component
                + 0.15 * atr_contraction
                + 0.10 * range_tightness
                + 0.05 * volume_score
            )
        )

        rows.append(
            {
                "ticker": ticker,
                "waiting_score": waiting_score,
                "rs_percentile": rs_percentile,
                "distance_to_breakout_pct": distance_to_breakout_pct,
                "breakout_level": float(resistance),
                "current_close": current_close,
                "current_high": current_high,
                "volume_ratio": volume_ratio,
                "atr_contraction": atr_contraction,
                "range_tightness": range_tightness,
                "strict_ready": strict_ready,
            }
        )

    waiting = pd.DataFrame(rows)
    if waiting.empty:
        return waiting
    return waiting.sort_values(["waiting_score", "ticker"], ascending=[False, True]).reset_index(drop=True)


def analyze_ticker_breakouts(price_df: pd.DataFrame, ticker: str, params: BreakoutParams) -> list[BreakoutEvent]:
    frame = _prepare_frame(price_df)
    min_required_rows = max(params.lookback_high, params.vol_window, 20) + params.hold_days + 2
    if frame.empty or len(frame) < min_required_rows:
        return []

    resistance = frame["Close"].rolling(params.lookback_high).max().shift(1)
    avg_volume = frame["Volume"].rolling(params.vol_window).mean().shift(1)
    prior_breakout = frame["Close"].shift(1) > resistance.shift(1)
    tr = _true_range(frame)
    atr_5 = tr.rolling(5).mean().shift(1)
    atr_20 = tr.rolling(20).mean().shift(1)
    atr_pct_5 = atr_5 / frame["Close"].shift(1)
    atr_pct_20 = atr_20 / frame["Close"].shift(1)
    close_range_5 = (frame["Close"].rolling(5).max() - frame["Close"].rolling(5).min()).shift(1)
    close_range_20 = (frame["Close"].rolling(20).max() - frame["Close"].rolling(20).min()).shift(1)
    prior_close = frame["Close"].shift(1)
    volume_ratio = frame["Volume"] / avg_volume

    breakout_mask = (
        frame["Close"].gt(resistance)
        & volume_ratio.ge(params.volume_multiplier)
        & resistance.notna()
        & avg_volume.notna()
        & ~prior_breakout.fillna(False)
    )

    if params.strict_setup:
        strict_mask = (
            atr_pct_5.lt(atr_pct_20)
            & close_range_5.le(close_range_20 * 0.60)
            & prior_close.ge(resistance * 0.98)
            & atr_pct_5.notna()
            & atr_pct_20.notna()
            & close_range_5.notna()
            & close_range_20.notna()
        )
        breakout_mask &= strict_mask

    breakout_positions = np.flatnonzero(breakout_mask.to_numpy(dtype=bool))
    events: list[BreakoutEvent] = []
    for position in breakout_positions:
        breakout_date = pd.Timestamp(frame.index[position])
        breakout_price = float(resistance.iloc[position])
        breakout_close = float(frame["Close"].iloc[position])
        future = frame.iloc[position + 1 : position + 1 + params.hold_days]
        stop_loss_price = breakout_price * (1.0 - params.stop_loss_pct)
        ret_5d = None
        if len(future) == params.hold_days:
            ret_5d = float(future["Close"].iloc[-1] / breakout_close - 1.0)

        if len(future) < params.hold_days:
            events.append(
                BreakoutEvent(
                    ticker=ticker,
                    breakout_date=breakout_date,
                    breakout_price=breakout_price,
                    outcome="pending",
                    failure_reason=None,
                    return_5d=ret_5d,
                    volume_ratio=float(volume_ratio.iloc[position]),
                    breakout_close=breakout_close,
                    stop_loss_price=stop_loss_price,
                    resolved_at=None,
                    resolution_days=None,
                    hold_days=params.hold_days,
                    strict_setup=params.strict_setup,
                    atr_pct_5=_float_or_none(atr_pct_5.iloc[position]),
                    atr_pct_20=_float_or_none(atr_pct_20.iloc[position]),
                    close_range_5=_float_or_none(close_range_5.iloc[position]),
                    close_range_20=_float_or_none(close_range_20.iloc[position]),
                    previous_close=_float_or_none(prior_close.iloc[position]),
                )
            )
            continue

        close_failure = future["Close"] < breakout_price
        stop_failure = future["Low"] <= stop_loss_price
        close_failure_date = future.index[close_failure.argmax()] if close_failure.any() else None
        stop_failure_date = future.index[stop_failure.argmax()] if stop_failure.any() else None

        outcome = "success"
        failure_reason = None
        resolved_at = pd.Timestamp(future.index[-1])
        resolution_days = params.hold_days
        if close_failure.any() or stop_failure.any():
            failure_dates = [item for item in [close_failure_date, stop_failure_date] if item is not None]
            earliest_failure = pd.Timestamp(min(failure_dates))
            outcome = "failure"
            resolved_at = earliest_failure
            resolution_days = int(future.index.get_loc(earliest_failure)) + 1
            if stop_failure_date is not None and pd.Timestamp(stop_failure_date) <= earliest_failure:
                failure_reason = config.FAILURE_STOP_LOSS
            else:
                failure_reason = config.FAILURE_CLOSE_LOST
        elif future["Close"].iloc[-1] < breakout_close or not future["Close"].gt(breakout_price).all():
            outcome = "failure"
            failure_reason = config.FAILURE_CLOSE_LOST

        events.append(
            BreakoutEvent(
                ticker=ticker,
                breakout_date=breakout_date,
                breakout_price=breakout_price,
                outcome=outcome,
                failure_reason=failure_reason,
                return_5d=ret_5d,
                volume_ratio=float(volume_ratio.iloc[position]),
                breakout_close=breakout_close,
                stop_loss_price=stop_loss_price,
                resolved_at=resolved_at,
                resolution_days=resolution_days,
                hold_days=params.hold_days,
                strict_setup=params.strict_setup,
                atr_pct_5=_float_or_none(atr_pct_5.iloc[position]),
                atr_pct_20=_float_or_none(atr_pct_20.iloc[position]),
                close_range_5=_float_or_none(close_range_5.iloc[position]),
                close_range_20=_float_or_none(close_range_20.iloc[position]),
                previous_close=_float_or_none(prior_close.iloc[position]),
            )
        )
    return events


def _snapshot_from_rows(rows: pd.DataFrame, as_of: pd.Timestamp) -> BreadthSnapshot:
    successes = int((rows["outcome"] == "success").sum()) if not rows.empty else 0
    failures = int((rows["outcome"] == "failure").sum()) if not rows.empty else 0
    pending = int((rows["outcome"] == "pending").sum()) if not rows.empty else 0
    attempts = successes + failures
    breadth_index = float(((successes - failures) / attempts) * 100.0) if attempts else 0.0
    success_rate = float(successes / attempts) if attempts else 0.0
    regime = _infer_regime(attempts, breadth_index, success_rate)
    return BreadthSnapshot(
        as_of=pd.Timestamp(as_of),
        attempts=attempts,
        successes=successes,
        failures=failures,
        pending=pending,
        breadth_index=breadth_index,
        regime=regime,
        confidence_flag=attempts < config.LOW_CONFIDENCE_ATTEMPTS,
    )


def _empty_analysis(params: BreakoutParams, as_of: pd.Timestamp) -> UniverseBreadthAnalysis:
    snapshot = _snapshot_from_rows(pd.DataFrame(columns=["outcome"]), as_of)
    history = pd.DataFrame(
        [
            {
                "as_of": as_of,
                "attempts": snapshot.attempts,
                "successes": snapshot.successes,
                "failures": snapshot.failures,
                "pending": snapshot.pending,
                "breadth_index": snapshot.breadth_index,
                "regime": snapshot.regime,
                "confidence_flag": snapshot.confidence_flag,
                "success_rate": 0.0,
            }
        ]
    )
    return UniverseBreadthAnalysis(
        params=params,
        snapshot=snapshot,
        history=history,
        daily_counts=pd.DataFrame(columns=["breakout_date", "success", "failure", "pending"]),
        watchlist=pd.DataFrame(),
        rs_ranking=pd.DataFrame(),
        waiting_watchlist=pd.DataFrame(),
        last_events=pd.DataFrame(),
        events=tuple(),
        diagnostics={"tickers_requested": 0, "tickers_analyzed": 0, "tickers_with_events": 0, "event_count": 0, "window_start": as_of, "window_end": as_of},
    )


def analyze_universe_breakouts(
    price_frames: Mapping[str, pd.DataFrame],
    params: BreakoutParams,
    *,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
    benchmark_frame: pd.DataFrame | None = None,
) -> UniverseBreadthAnalysis:
    valid_items = [(ticker, _prepare_frame(frame)) for ticker, frame in price_frames.items() if frame is not None]
    valid_items = [(ticker, frame) for ticker, frame in valid_items if not frame.empty]
    if not valid_items:
        as_of = pd.Timestamp(end_date) if end_date is not None else pd.Timestamp.utcnow().normalize()
        return _empty_analysis(params, as_of)

    calendar = pd.DatetimeIndex(sorted({stamp for _, frame in valid_items for stamp in frame.index}))
    if calendar.empty:
        as_of = pd.Timestamp(end_date) if end_date is not None else pd.Timestamp.utcnow().normalize()
        return _empty_analysis(params, as_of)

    effective_end = pd.Timestamp(end_date) if end_date is not None else pd.Timestamp(calendar.max())
    calendar = calendar[calendar <= effective_end]
    if calendar.empty:
        return _empty_analysis(params, effective_end)

    if start_date is not None:
        visible_calendar = calendar[calendar >= pd.Timestamp(start_date)]
    else:
        visible_calendar = calendar[-params.scan_window :]
    if visible_calendar.empty:
        return _empty_analysis(params, effective_end)

    max_workers = min(config.MAX_ANALYSIS_WORKERS, max(1, os.cpu_count() or 1), max(1, len(valid_items)))
    all_events: list[BreakoutEvent] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for ticker_events in executor.map(lambda item: analyze_ticker_breakouts(item[1], item[0], params), valid_items):
            all_events.extend(ticker_events)

    rows = pd.DataFrame(_event_to_row(event) for event in all_events)
    if rows.empty:
        latest_as_of = pd.Timestamp(visible_calendar.max())
        rs_ranking = _build_relative_strength_table(valid_items, benchmark_frame, latest_as_of)
        waiting_watchlist = _build_waiting_watchlist(
            valid_items,
            params,
            latest_as_of,
            rs_ranking,
        )
        return UniverseBreadthAnalysis(
            params=params,
            snapshot=_snapshot_from_rows(pd.DataFrame(columns=["outcome"]), latest_as_of),
            history=pd.DataFrame(
                [
                    {
                        "as_of": latest_as_of,
                        "attempts": 0,
                        "successes": 0,
                        "failures": 0,
                        "pending": 0,
                        "breadth_index": 0.0,
                        "regime": _infer_regime(0, 0.0, 0.0),
                        "confidence_flag": True,
                        "success_rate": 0.0,
                    }
                ]
            ),
            daily_counts=pd.DataFrame(columns=["breakout_date", "success", "failure", "pending"]),
            watchlist=pd.DataFrame(),
            rs_ranking=rs_ranking,
            waiting_watchlist=waiting_watchlist,
            last_events=pd.DataFrame(),
            events=tuple(),
            diagnostics={
                "tickers_requested": len(price_frames),
                "tickers_analyzed": len(valid_items),
                "tickers_with_events": 0,
                "event_count": 0,
                "window_start": latest_as_of,
                "window_end": latest_as_of,
                "max_workers": max_workers,
                "benchmark_symbol": config.BENCHMARK_SYMBOL,
                "benchmark_available": bool(benchmark_frame is not None and not benchmark_frame.empty),
            },
        )

    rows["breakout_date"] = pd.to_datetime(rows["breakout_date"])
    rows["resolved_at"] = pd.to_datetime(rows["resolved_at"])
    rows = rows.sort_values(["breakout_date", "ticker"]).reset_index(drop=True)

    history_rows: list[dict[str, Any]] = []
    visible_dates = pd.DatetimeIndex(visible_calendar)
    full_calendar_lookup = pd.Series(range(len(calendar)), index=calendar)
    for as_of in visible_dates:
        full_position = int(full_calendar_lookup.loc[pd.Timestamp(as_of)])
        start_position = max(0, full_position - params.scan_window + 1)
        window_dates = calendar[start_position : full_position + 1]
        window_start = pd.Timestamp(window_dates.min())
        window_rows = rows.loc[(rows["breakout_date"] >= window_start) & (rows["breakout_date"] <= as_of)].copy()
        if not window_rows.empty:
            unresolved = window_rows["resolved_at"].isna() | (window_rows["resolved_at"] > as_of)
            window_rows.loc[unresolved, "outcome"] = "pending"
        snapshot = _snapshot_from_rows(window_rows, pd.Timestamp(as_of))
        success_rate = float(snapshot.successes / snapshot.attempts) if snapshot.attempts else 0.0
        history_rows.append(
            {
                "as_of": pd.Timestamp(as_of),
                "attempts": snapshot.attempts,
                "successes": snapshot.successes,
                "failures": snapshot.failures,
                "pending": snapshot.pending,
                "breadth_index": snapshot.breadth_index,
                "regime": snapshot.regime,
                "confidence_flag": snapshot.confidence_flag,
                "success_rate": success_rate,
            }
        )

    history = pd.DataFrame(history_rows)
    latest_as_of = pd.Timestamp(visible_dates.max())
    latest_full_position = int(full_calendar_lookup.loc[latest_as_of])
    latest_window_dates = calendar[max(0, latest_full_position - params.scan_window + 1) : latest_full_position + 1]
    latest_window_start = pd.Timestamp(latest_window_dates.min())
    latest_rows = rows.loc[(rows["breakout_date"] >= latest_window_start) & (rows["breakout_date"] <= latest_as_of)].copy()
    if not latest_rows.empty:
        unresolved = latest_rows["resolved_at"].isna() | (latest_rows["resolved_at"] > latest_as_of)
        latest_rows.loc[unresolved, "outcome"] = "pending"
    latest_snapshot = _snapshot_from_rows(latest_rows, latest_as_of)

    in_window = rows.loc[(rows["breakout_date"] >= latest_window_start) & (rows["breakout_date"] <= latest_as_of)].copy()
    if not in_window.empty:
        unresolved = in_window["resolved_at"].isna() | (in_window["resolved_at"] > latest_as_of)
        in_window.loc[unresolved, "outcome"] = "pending"
    daily_counts = (
        in_window.groupby(["breakout_date", "outcome"]).size().unstack(fill_value=0).reset_index().rename_axis(None, axis=1)
        if not in_window.empty
        else pd.DataFrame(columns=["breakout_date", "success", "failure", "pending"])
    )
    for column in config.EVENT_OUTCOMES:
        if column not in daily_counts.columns:
            daily_counts[column] = 0
    if not daily_counts.empty:
        daily_counts = daily_counts.loc[:, ["breakout_date", "success", "failure", "pending"]].sort_values("breakout_date")

    watchlist = in_window.loc[in_window["outcome"] == "failure"].copy()
    if not watchlist.empty:
        watchlist["failure_label"] = watchlist["failure_reason"].map(failure_reason_label)
        watchlist = watchlist.sort_values(["resolved_at", "breakout_date", "ticker"], ascending=[False, False, True]).reset_index(drop=True)

    last_events = in_window.sort_values(["breakout_date", "ticker"]).groupby("ticker", as_index=False).tail(1).reset_index(drop=True)
    if not last_events.empty:
        last_events = last_events.sort_values(["breakout_date", "ticker"], ascending=[False, True]).reset_index(drop=True)

    rs_ranking = _build_relative_strength_table(valid_items, benchmark_frame, latest_as_of)
    waiting_watchlist = _build_waiting_watchlist(valid_items, params, latest_as_of, rs_ranking)

    diagnostics = {
        "tickers_requested": len(price_frames),
        "tickers_analyzed": len(valid_items),
        "tickers_with_events": int(rows["ticker"].nunique()),
        "event_count": int(len(rows)),
        "window_start": latest_window_start,
        "window_end": latest_as_of,
        "max_workers": max_workers,
        "benchmark_symbol": config.BENCHMARK_SYMBOL,
        "benchmark_available": bool(benchmark_frame is not None and not benchmark_frame.empty),
    }
    return UniverseBreadthAnalysis(
        params=params,
        snapshot=latest_snapshot,
        history=history,
        daily_counts=daily_counts,
        watchlist=watchlist,
        rs_ranking=rs_ranking,
        waiting_watchlist=waiting_watchlist,
        last_events=last_events,
        events=tuple(all_events),
        diagnostics=diagnostics,
    )
