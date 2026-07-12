"""Write-only JSON audit exports for the agentic tool-use demo."""

from __future__ import annotations

import datetime
import inspect
import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from hf_router import LocalHFClassificationRouter, select_tool_from_scores
from scorers import CALIBRATION_USER_REQUESTS

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

LOGGER = logging.getLogger(__name__)
EXPORT_DIR = Path(__file__).resolve().parent / "exports"
TOOL_SCORE_NAMES = ("weather_tool", "calculator_tool", "web_search_tool", "no_tool")


def routing_defaults() -> dict[str, object]:
    """Read the live HF routing defaults from their defining signatures."""
    router_parameters = inspect.signature(LocalHFClassificationRouter.__init__).parameters
    selection_parameters = inspect.signature(select_tool_from_scores).parameters
    return {
        "selection_mode": router_parameters["selection_mode"].default,
        "no_tool_boost_delta": float(selection_parameters["no_tool_boost_delta"].default),
    }


def _timestamp(now: datetime.datetime | None = None) -> datetime.datetime:
    value = now or datetime.datetime.now(tz=datetime.UTC)
    return value if value.tzinfo is not None else value.replace(tzinfo=datetime.UTC)


def _json_native(value: object) -> object:
    """Recursively convert numpy scalars and containers to JSON-native values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_native(item) for item in value]
    return value


def _write_json(payload: Mapping[str, object], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(_json_native(dict(payload)), file, indent=2, ensure_ascii=False)
    return path


def write_config_snapshot(
    *,
    hf_model_id: str,
    device: object,
    export_dir: Path = EXPORT_DIR,
    now: datetime.datetime | None = None,
) -> Path:
    """Write a read-only audit snapshot of the live HF routing configuration."""
    created_at = _timestamp(now)
    defaults = routing_defaults()
    payload = {
        "config_version": "1.0",
        "created_at": created_at.isoformat(),
        "model": {"hf_model_id": hf_model_id, "device": str(device)},
        "routing": {
            "selection_mode": defaults["selection_mode"],
            "no_tool_boost_delta": defaults["no_tool_boost_delta"],
            "raw_margin_threshold": None,
            "calibration_strength": None,
        },
        "calibration": {"calibration_user_requests": list(CALIBRATION_USER_REQUESTS)},
    }
    filename = f"config_{created_at.strftime('%Y%m%d_%H%M%S')}.json"
    return _write_json(payload, export_dir / filename)


def write_config_snapshot_safely(**kwargs: object) -> Path | None:
    """Write a config snapshot without allowing audit I/O to break model loading."""
    try:
        return write_config_snapshot(**kwargs)  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001
        LOGGER.warning("Could not write the agentic config snapshot.", exc_info=True)
        return None


def _request_slug(user_request: str, max_length: int = 30) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", user_request.lower()).strip("_")
    return slug[:max_length].rstrip("_") or "request"


def build_pairwise_interactions(
    *,
    player_segments: Sequence[object],
    pairwise_matrix: object,
) -> list[dict[str, object]]:
    """Convert the displayed upper-triangle k-SII matrix into JSON rows."""
    rows: list[dict[str, object]] = []
    for left_index, left_segment in enumerate(player_segments):
        for right_index in range(left_index + 1, len(player_segments)):
            right_segment = player_segments[right_index]
            value = pairwise_matrix.iloc[left_index, right_index]
            rows.append(
                {
                    "pair": [left_segment.label, right_segment.label],
                    "text": [left_segment.text, right_segment.text],
                    "k_sii": float(value),
                }
            )
    return rows


def build_result_payload(
    *,
    hf_model_id: str,
    user_request: str,
    system_prompt: str,
    player_segments: Sequence[object],
    raw_scores: Mapping[str, object],
    calibrated_scores: Mapping[str, object],
    selected_tool: str | None,
    raw_argmax: str | None,
    calibrated_argmax: str | None,
    target_tool: str,
    baseline_h_empty: object,
    full_h_n: object,
    pairwise_interactions: Sequence[Mapping[str, object]],
    now: datetime.datetime | None = None,
) -> tuple[dict[str, object], datetime.datetime]:
    """Build one JSON-native routing and XAI result envelope."""
    created_at = _timestamp(now)
    defaults = routing_defaults()
    empty_value = float(baseline_h_empty)
    full_value = float(full_h_n)
    payload = {
        "object_name": "AgenticToolUseResult",
        "data_type": "routing_and_xai_result",
        "version": "1.0",
        "timestamp": created_at.isoformat(),
        "config": {
            "hf_model_id": hf_model_id,
            "selection_mode": defaults["selection_mode"],
            "no_tool_boost_delta": defaults["no_tool_boost_delta"],
        },
        "request": {
            "user_request": user_request,
            "system_prompt": system_prompt,
            "player_segments": [segment.text for segment in player_segments],
        },
        "routing_decision": {
            "raw_scores": {name: float(value) for name, value in raw_scores.items()},
            "calibrated_scores": {name: float(value) for name, value in calibrated_scores.items()},
            "selected_tool": selected_tool,
            "raw_argmax": raw_argmax,
            "calibrated_argmax": calibrated_argmax,
        },
        "xai_explanation": {
            "target_tool": target_tool,
            "baseline_h_empty": empty_value,
            "full_h_n": full_value,
            "delta": float(full_value - empty_value),
            "pairwise_interactions": list(pairwise_interactions),
        },
    }
    return _json_native(payload), created_at  # type: ignore[return-value]


def write_result_export(
    *,
    export_dir: Path = EXPORT_DIR,
    **payload_kwargs: Any,
) -> Path:
    """Build and write one completed pipeline result export."""
    payload, created_at = build_result_payload(**payload_kwargs)
    slug = _request_slug(str(payload["request"]["user_request"]))  # type: ignore[index]
    filename = f"result_{created_at.strftime('%Y%m%d_%H%M%S')}_{slug}.json"
    return _write_json(payload, export_dir / filename)


def write_result_export_safely(
    *,
    warning_callback: Callable[[str], object] | None = None,
    **kwargs: Any,
) -> Path | None:
    """Write a result export without allowing optional I/O to break a run."""
    try:
        return write_result_export(**kwargs)
    except Exception as error:  # noqa: BLE001
        message = f"Could not write the run export: {error}"
        LOGGER.warning(message, exc_info=True)
        if warning_callback is not None:
            warning_callback(message)
        return None
