"""Reliable, fail-closed HTTP boundary for weather data providers."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any, Callable

import requests


RETRYABLE_HTTP_STATUSES = frozenset({429, 500, 502, 503, 504})


class WeatherDataUnavailable(RuntimeError):
    """Raised when a weather provider cannot return trustworthy JSON."""


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_seconds: float = 0.75
    max_delay_seconds: float = 6.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.base_delay_seconds < 0 or self.max_delay_seconds < 0:
            raise ValueError("retry delays cannot be negative")


class WeatherHttpClient:
    """Small synchronous JSON client with bounded, provider-aware retries.

    Weatherbet's scan path is synchronous. Keeping this client synchronous avoids
    mixing blocking requests with an event loop while still handling transient
    DNS, connection, timeout, rate-limit, and upstream-service failures.
    """

    def __init__(
        self,
        *,
        user_agent: str,
        retry_policy: RetryPolicy | None = None,
        request_get: Callable[..., Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        random_uniform: Callable[[float, float], float] = random.uniform,
    ) -> None:
        self._headers = {
            "Accept": "application/json",
            "User-Agent": user_agent,
        }
        self._retry_policy = retry_policy or RetryPolicy()
        self._request_get = request_get
        self._sleep = sleep
        self._random_uniform = random_uniform

    def get_json(
        self,
        url: str,
        *,
        provider: str,
        timeout: tuple[float, float],
    ) -> Any | None:
        """Return decoded JSON, ``None`` for HTTP 204, or fail closed."""
        last_error = "unknown provider failure"
        policy = self._retry_policy

        for attempt in range(policy.max_attempts):
            response = None
            try:
                request_get = self._request_get or requests.get
                response = request_get(
                    url,
                    headers=self._headers,
                    timeout=timeout,
                )
                status = int(getattr(response, "status_code", 200))
                if status == 204:
                    return None
                if status in RETRYABLE_HTTP_STATUSES:
                    last_error = f"HTTP {status}"
                elif status >= 400:
                    detail = self._error_detail(response)
                    suffix = f" ({detail})" if detail else ""
                    raise WeatherDataUnavailable(f"{provider}: HTTP {status}{suffix}")
                else:
                    try:
                        return response.json()
                    except (TypeError, ValueError) as exc:
                        last_error = f"invalid or empty JSON ({exc})"
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            except requests.RequestException as exc:
                raise WeatherDataUnavailable(f"{provider}: {exc}") from exc

            if attempt + 1 < policy.max_attempts:
                self._sleep(self._retry_delay(attempt, response))

        raise WeatherDataUnavailable(
            f"{provider}: unavailable after {policy.max_attempts} attempts ({last_error})"
        )

    def _retry_delay(self, attempt: int, response: Any | None) -> float:
        retry_after = self._retry_after_seconds(response)
        if retry_after is not None:
            return min(retry_after, self._retry_policy.max_delay_seconds)

        exponential = self._retry_policy.base_delay_seconds * (2**attempt)
        jitter = self._random_uniform(0.0, self._retry_policy.base_delay_seconds)
        return min(exponential + jitter, self._retry_policy.max_delay_seconds)

    @staticmethod
    def _error_detail(response: Any) -> str:
        """Extract a bounded provider reason without leaking full responses."""
        try:
            payload = response.json()
        except (TypeError, ValueError):
            return ""
        if not isinstance(payload, dict):
            return ""
        detail = payload.get("reason") or payload.get("message") or payload.get("error")
        if detail is None or isinstance(detail, (dict, list)):
            return ""
        return str(detail).strip()[:240]

    @staticmethod
    def _retry_after_seconds(response: Any | None) -> float | None:
        if response is None:
            return None
        value = getattr(response, "headers", {}).get("Retry-After")
        if not value:
            return None
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            try:
                retry_at = parsedate_to_datetime(str(value))
                return max(0.0, retry_at.timestamp() - time.time())
            except (TypeError, ValueError, OverflowError):
                return None
