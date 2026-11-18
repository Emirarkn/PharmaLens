"""Time-series forecasts on GLN-total series.

Holt-Winters (additive level/trend/seasonality, period 12) vectorised across all pharmacies,
with one global smoothing-parameter set tuned on train-period validation only. Unobserved
months do not update the state (the one-step forecast is carried), so no value is invented.
"""
from __future__ import annotations

import itertools

import numpy as np


def holt_winters(Y: np.ndarray, alpha: float, beta: float, gamma: float, h: int = 3, period: int = 12) -> np.ndarray:
    """Forecast the next ``h`` months for each row of ``Y`` ([G, T] with NaN). Returns [G, h] (>= 0)."""
    G, T = Y.shape
    if T < period + 2:
        base = np.nanmean(Y[:, -3:], axis=1) if T else np.full(G, np.nan)
        return np.repeat(np.clip(np.nan_to_num(base), 0, None)[:, None], h, axis=1)
    with np.errstate(all="ignore"):
        first = Y[:, :period]
        level = np.nanmean(first, axis=1)
        level = np.where(np.isnan(level), 0.0, level)
        season = np.nan_to_num(first - level[:, None])
        trend = np.zeros(G)
    for t in range(period, T):
        s_idx = t % period
        fc = level + trend + season[:, s_idx]
        y = Y[:, t]
        obs = ~np.isnan(y)
        y_eff = np.where(obs, y, fc)
        new_level = np.where(obs, alpha * (y_eff - season[:, s_idx]) + (1 - alpha) * (level + trend), level + trend)
        trend = np.where(obs, beta * (new_level - level) + (1 - beta) * trend, trend)
        season[:, s_idx] = np.where(obs, gamma * (y_eff - new_level) + (1 - gamma) * season[:, s_idx], season[:, s_idx])
        level = new_level
    out = np.stack([level + (k + 1) * trend + season[:, (T + k) % period] for k in range(h)], axis=1)
    return np.clip(out, 0, None)


def seasonal_naive(Y: np.ndarray, h: int = 3, period: int = 12) -> np.ndarray:
    """Same months last year; falls back to the recent mean where last year is unobserved."""
    G, T = Y.shape
    recent = np.nanmean(Y[:, -3:], axis=1) if T >= 1 else np.full(G, np.nan)
    out = np.empty((G, h))
    for k in range(h):
        j = T + k - period
        last_year = Y[:, j] if 0 <= j < T else np.full(G, np.nan)
        out[:, k] = np.where(np.isnan(last_year), recent, last_year)
    return out


def moving_average(Y: np.ndarray, h: int = 3, window: int = 3) -> np.ndarray:
    with np.errstate(all="ignore"):
        m = np.nanmean(Y[:, -window:], axis=1)
    return np.repeat(m[:, None], h, axis=1)


def wape(actual: np.ndarray, pred: np.ndarray) -> float | None:
    ok = ~np.isnan(actual) & ~np.isnan(pred)
    den = np.abs(actual[ok]).sum()
    return None if den == 0 else float(np.abs(actual[ok] - pred[ok]).sum() / den)


def tune_holt_winters(Y: np.ndarray, origins_idx: list[int], grid: dict[str, list[float]], h: int = 3) -> dict:
    """Pick (alpha, beta, gamma) minimising pooled WAPE of 3-month sums over the given train origins."""
    best, results = None, []
    for a, b, g in itertools.product(grid["alpha"], grid["beta"], grid["gamma"]):
        actual, pred = [], []
        for o in origins_idx:
            if o + h >= Y.shape[1]:
                continue
            fut = Y[:, o + 1:o + 1 + h]
            ok = ~np.isnan(fut).any(axis=1)
            fc = holt_winters(Y[:, :o + 1], a, b, g, h)
            actual.append(fut[ok].sum(axis=1))
            pred.append(fc[ok].sum(axis=1))
        if not actual:
            continue
        score = wape(np.concatenate(actual), np.concatenate(pred))
        results.append({"alpha": a, "beta": b, "gamma": g, "wape": score})
        if score is not None and (best is None or score < best["wape"]):
            best = results[-1]
    return {"best": best or {"alpha": 0.3, "beta": 0.0, "gamma": 0.2, "wape": None}, "grid": results}
