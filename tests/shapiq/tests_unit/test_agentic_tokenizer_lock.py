"""Regression tests for concurrent access to shared HF tokenizers."""

from __future__ import annotations

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DEMO_DIR = Path(__file__).parents[3] / "src" / "demos" / "agentic_tool_use_explanation"
sys.path.insert(0, str(DEMO_DIR))

from hf_router import HFArgumentExtractor  # noqa: E402
from scorers import NativeToolCallScorer  # noqa: E402


class BorrowCheckingTokenizer:
    """Fail deterministically when two threads overlap one tokenizer call."""

    def __init__(self) -> None:
        self._state_lock = threading.Lock()
        self._in_use = False

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        del add_special_tokens
        with self._state_lock:
            if self._in_use:
                msg = "Already borrowed"
                raise RuntimeError(msg)
            self._in_use = True
        try:
            time.sleep(0.02)
            return {"input_ids": list(range(max(1, len(text))))}
        finally:
            with self._state_lock:
                self._in_use = False


def _run_concurrently(callable_under_test) -> list[BaseException]:
    barrier = threading.Barrier(8)

    def invoke() -> None:
        barrier.wait()
        callable_under_test()

    errors: list[BaseException] = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(invoke) for _ in range(8)]
        for future in futures:
            try:
                future.result()
            except BaseException as error:  # noqa: BLE001 - errors are the test result
                errors.append(error)
    return errors


def test_shared_tokenizer_lock_prevents_already_borrowed_race() -> None:
    unlocked_tokenizer = BorrowCheckingTokenizer()
    unlocked_errors = _run_concurrently(
        lambda: unlocked_tokenizer("prompt", add_special_tokens=False)
    )
    assert len(unlocked_errors) == 7
    assert all(str(error) == "Already borrowed" for error in unlocked_errors)

    locked_tokenizer = BorrowCheckingTokenizer()
    scorer = object.__new__(NativeToolCallScorer)
    scorer.tokenizer = locked_tokenizer
    scorer.tokenizer_lock = threading.RLock()

    locked_errors = _run_concurrently(
        lambda: scorer._continuation_token_count("prompt", " continuation")
    )
    assert locked_errors == []


def test_native_wrappers_reuse_the_cached_resource_lock() -> None:
    class FakeModel:
        def eval(self) -> None:
            return None

    class FakeTokenizer:
        pad_token = "<pad>"
        eos_token = "<eos>"
        padding_side = "left"

    shared_lock = threading.RLock()
    model = FakeModel()
    tokenizer = FakeTokenizer()

    scorer = NativeToolCallScorer(
        model=model,
        tokenizer=tokenizer,
        device="cpu",
        tokenizer_lock=shared_lock,
    )
    extractor = HFArgumentExtractor(
        model=model,
        tokenizer=tokenizer,
        device="cpu",
        tokenizer_lock=shared_lock,
    )

    assert scorer.tokenizer_lock is shared_lock
    assert extractor.tokenizer_lock is shared_lock
