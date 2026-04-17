from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, timedelta
import hashlib
from io import StringIO
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd
import requests

from . import config

try:
    import yfinance as yf
except Exception:
    yf = None


@dataclass(frozen=True)
class ProviderDiagnostics:
    provider_name: str
    requested_tickers: int
    returned_tickers: int
    missing_tickers: tuple[str, ...]
    batch_count: int
    cache_hits: int
    start_date: str | None
    end_date: str | None
    period: str | None


class PriceProvider(ABC):
    provider_name: str

    @abstractmethod
    def download_batch(
        self,
        tickers: list[str],
        *,
        start_date: date | None = None,
        end_date: date | None = None,
        period: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        raise NotImplementedError


def _ensure_cache_dirs() -> None:
    config.CONSTITUENT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    config.OHLCV_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def clear_all_caches() -> None:
    if config.CACHE_ROOT.exists():
        shutil.rmtree(config.CACHE_ROOT)


def normalize_symbol_for_yahoo(symbol: str) -> str:
    return symbol.strip().upper().replace(".", "-")


def _constituent_cache_path() -> Path:
    _ensure_cache_dirs()
    return config.CONSTITUENT_CACHE_DIR / f"sp500_constituents_{date.today().isoformat()}.pkl"


def _latest_constituent_cache() -> Path | None:
    _ensure_cache_dirs()
    cache_files = sorted(config.CONSTITUENT_CACHE_DIR.glob("sp500_constituents_*.pkl"))
    if not cache_files:
        return None
    return cache_files[-1]


def _extract_constituent_table(tables: list[pd.DataFrame]) -> pd.DataFrame:
    for table in tables:
        if {"Symbol", "Security"}.issubset(table.columns):
            selected = [column for column in ["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"] if column in table.columns]
            frame = table.loc[:, selected].copy()
            frame["Symbol"] = frame["Symbol"].astype(str).str.upper()
            frame["YahooSymbol"] = frame["Symbol"].map(normalize_symbol_for_yahoo)
            return frame.drop_duplicates(subset=["YahooSymbol"]).reset_index(drop=True)
    raise RuntimeError("Wikipedia did not return a usable S&P 500 constituents table.")


def load_sp500_constituents(force_refresh: bool = False) -> pd.DataFrame:
    cache_path = _constituent_cache_path()
    if cache_path.exists() and not force_refresh:
        return pd.read_pickle(cache_path)

    try:
        response = requests.get(
            config.WIKI_SP500_URL,
            headers={"User-Agent": "Mozilla/5.0 breakout-breadth-dashboard"},
            timeout=config.REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text))
        frame = _extract_constituent_table(tables)
        frame.to_pickle(cache_path)
        return frame
    except Exception:
        fallback_cache = _latest_constituent_cache()
        if fallback_cache is not None:
            return pd.read_pickle(fallback_cache)
        raise


class YahooFinanceProvider(PriceProvider):
    provider_name = "yfinance"

    def download_batch(
        self,
        tickers: list[str],
        *,
        start_date: date | None = None,
        end_date: date | None = None,
        period: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        if yf is None:
            raise RuntimeError("yfinance is not installed in the active environment.")

        kwargs: dict[str, Any] = {
            "progress": False,
            "auto_adjust": False,
            "threads": False,
            "group_by": "column",
        }
        if start_date is not None:
            kwargs["start"] = start_date.isoformat()
            if end_date is not None:
                kwargs["end"] = (end_date + timedelta(days=1)).isoformat()
        elif period is not None:
            kwargs["period"] = period
        raw = yf.download(tickers, **kwargs)
        return _split_download_frame(raw, tickers)


class FinancialModelingPrepProvider(PriceProvider):
    provider_name = "fmp"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def download_batch(
        self,
        tickers: list[str],
        *,
        start_date: date | None = None,
        end_date: date | None = None,
        period: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        frames: dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            frames[ticker] = self._download_single(ticker, start_date=start_date, end_date=end_date)
        return frames

    def _download_single(self, ticker: str, *, start_date: date | None, end_date: date | None) -> pd.DataFrame:
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}"
        params = {"apikey": self.api_key}
        if start_date is not None:
            params["from"] = start_date.isoformat()
        if end_date is not None:
            params["to"] = end_date.isoformat()
        response = requests.get(url, params=params, timeout=config.REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("historical", [])
        if not rows:
            return pd.DataFrame(columns=config.REQUIRED_OHLCV_COLUMNS)
        frame = pd.DataFrame(rows)
        frame = frame.rename(
            columns={"date": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
        )
        if not {"Date", "Open", "High", "Low", "Close", "Volume"}.issubset(frame.columns):
            return pd.DataFrame(columns=config.REQUIRED_OHLCV_COLUMNS)
        frame["Date"] = pd.to_datetime(frame["Date"])
        frame = frame.set_index("Date").sort_index()
        return frame.loc[:, list(config.REQUIRED_OHLCV_COLUMNS)]


def select_default_provider_name() -> str:
    return os.getenv("BREAKOUT_BREADTH_PROVIDER", config.DEFAULT_PROVIDER).strip().lower()


def get_price_provider(provider_name: str | None = None) -> PriceProvider:
    selected = (provider_name or select_default_provider_name()).strip().lower()
    if selected == "fmp":
        api_key = os.getenv("FMP_API_KEY")
        if not api_key:
            raise RuntimeError("FMP provider requested but FMP_API_KEY is not set.")
        return FinancialModelingPrepProvider(api_key)
    return YahooFinanceProvider()


def _normalize_ohlcv_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=config.REQUIRED_OHLCV_COLUMNS)
    normalized = frame.copy()
    normalized.index = pd.to_datetime(normalized.index)
    normalized = normalized.sort_index()
    if "Adj Close" in normalized.columns and "Close" not in normalized.columns:
        normalized = normalized.rename(columns={"Adj Close": "Close"})
    for column in config.REQUIRED_OHLCV_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = pd.NA
    normalized = normalized.loc[:, list(config.REQUIRED_OHLCV_COLUMNS)]
    normalized = normalized.dropna(subset=["Open", "High", "Low", "Close"])
    normalized["Volume"] = normalized["Volume"].fillna(0.0)
    return normalized


def _split_download_frame(raw: pd.DataFrame, tickers: list[str]) -> dict[str, pd.DataFrame]:
    if raw is None or raw.empty:
        return {ticker: pd.DataFrame(columns=config.REQUIRED_OHLCV_COLUMNS) for ticker in tickers}

    frames: dict[str, pd.DataFrame] = {}
    if isinstance(raw.columns, pd.MultiIndex):
        for ticker in tickers:
            if ticker not in raw.columns.get_level_values(-1):
                frames[ticker] = pd.DataFrame(columns=config.REQUIRED_OHLCV_COLUMNS)
                continue
            frames[ticker] = _normalize_ohlcv_frame(raw.xs(ticker, axis=1, level=-1, drop_level=True))
        return frames

    if len(tickers) == 1:
        return {tickers[0]: _normalize_ohlcv_frame(raw)}

    return {ticker: pd.DataFrame(columns=config.REQUIRED_OHLCV_COLUMNS) for ticker in tickers}


def _chunked(values: Iterable[str], size: int) -> list[list[str]]:
    chunk: list[str] = []
    chunks: list[list[str]] = []
    for value in values:
        chunk.append(value)
        if len(chunk) == size:
            chunks.append(chunk)
            chunk = []
    if chunk:
        chunks.append(chunk)
    return chunks


def _cache_key(
    provider_name: str,
    tickers: list[str],
    *,
    start_date: date | None,
    end_date: date | None,
    period: str | None,
) -> str:
    payload = "|".join(
        [
            provider_name,
            ",".join(sorted(tickers)),
            start_date.isoformat() if start_date is not None else "",
            end_date.isoformat() if end_date is not None else "",
            period or "",
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _cache_path(
    provider_name: str,
    tickers: list[str],
    *,
    start_date: date | None,
    end_date: date | None,
    period: str | None,
) -> Path:
    _ensure_cache_dirs()
    cache_key = _cache_key(provider_name, tickers, start_date=start_date, end_date=end_date, period=period)
    return config.OHLCV_CACHE_DIR / f"{provider_name}_{cache_key}.pkl"


def download_ohlcv_universe(
    tickers: list[str],
    *,
    provider_name: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    period: str | None = config.DEFAULT_HISTORY_PERIOD,
    batch_size: int = config.DEFAULT_BATCH_SIZE,
    force_refresh: bool = False,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[dict[str, pd.DataFrame], ProviderDiagnostics]:
    provider = get_price_provider(provider_name)
    normalized = [normalize_symbol_for_yahoo(ticker) for ticker in tickers]
    all_frames: dict[str, pd.DataFrame] = {}
    cache_hits = 0
    batch_count = 0
    batches = _chunked(normalized, batch_size)
    total_batches = len(batches)

    for batch_index, batch in enumerate(batches, start=1):
        batch_count += 1
        cache_path = _cache_path(
            provider.provider_name,
            batch,
            start_date=start_date,
            end_date=end_date,
            period=None if start_date is not None else period,
        )
        if cache_path.exists() and not force_refresh:
            cache_hits += 1
            batch_frames = pd.read_pickle(cache_path)
            source = "cache"
        else:
            batch_frames = provider.download_batch(
                batch,
                start_date=start_date,
                end_date=end_date,
                period=None if start_date is not None else period,
            )
            pd.to_pickle(batch_frames, cache_path)
            source = "download"
        all_frames.update(batch_frames)
        if progress_callback is not None:
            progress_callback(batch_index, total_batches, source)

    missing = tuple(sorted(ticker for ticker in normalized if ticker not in all_frames or all_frames[ticker].empty))
    diagnostics = ProviderDiagnostics(
        provider_name=provider.provider_name,
        requested_tickers=len(normalized),
        returned_tickers=len(normalized) - len(missing),
        missing_tickers=missing,
        batch_count=batch_count,
        cache_hits=cache_hits,
        start_date=start_date.isoformat() if start_date is not None else None,
        end_date=end_date.isoformat() if end_date is not None else None,
        period=None if start_date is not None else period,
    )
    return all_frames, diagnostics
