"""Real Groq-router-backed coalition value-function scorers for the agentic demo.

Unlike ``groq_agent.run_groq_tool_inference``, which performs the one real
agent decision (and may execute a demo tool / draft a final answer),
these scorers are meant to be called during shapiq's coalition sampling. They
only ask Groq for routing decisions and never execute tools or ask Groq to
draft final answers.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

try:
    from demos.agentic_tool_use_explanation.scorers import split_coalition_prompt
except ModuleNotFoundError:
    from scorers import split_coalition_prompt

DEFAULT_GROQ_ROUTER_MODEL_ID = "llama-3.1-8b-instant"
DEFAULT_GROQ_SOFT_VOTE_N_SAMPLES = 5
DEFAULT_GROQ_SOFT_VOTE_TEMPERATURE = 0.3
DEFAULT_GROQ_SOFT_VOTE_MAX_RETRIES = 2


@dataclass
class GroqDeterministicRouterScorer:
    """Coalition value function backed by a real Groq tool-routing decision.

    For each coalition prompt, asks Groq which decision candidate it would
    pick for the (possibly masked) user request, then returns ``1.0`` if Groq
    selected ``target_tool`` and ``0.0`` otherwise. Routing decisions are
    cached per exact prompt/candidate-set, since shapiq's preview step and
    coalition sampling can request the same prompt for several target tools.
    """

    model_name: str = DEFAULT_GROQ_ROUTER_MODEL_ID
    client_factory: Callable[[str], object] | None = None
    last_debug_outputs: list[dict[str, object]] = field(default_factory=list, init=False)
    _client: object | None = field(default=None, init=False, repr=False)
    _decision_cache: dict[tuple[str, tuple[str, ...]], tuple[str | None, str]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def score_batch(
        self,
        prompts: list[str],
        *,
        target_tool: str,
        tool_descriptions: dict[str, str],
    ) -> list[float]:
        """Return 1.0 for each prompt where Groq would pick target_tool, else 0.0."""
        if target_tool not in tool_descriptions:
            msg = f"Target tool {target_tool!r} is not a known decision candidate."
            raise ValueError(msg)

        self.last_debug_outputs = []
        scores = []
        for prompt in prompts:
            selected_tool, raw_output = self._route(prompt, tool_descriptions)
            score = 1.0 if selected_tool == target_tool else 0.0
            scores.append(score)
            self.last_debug_outputs.append(
                {
                    "target_tool": target_tool,
                    "selected_tool": selected_tool,
                    "raw_output": raw_output,
                    "final_score": score,
                    "prompt_preview": prompt[:240],
                }
            )
        return scores

    def build_scoring_prompt(
        self,
        prompt: str,
        *,
        target_tool: str,
        tool_descriptions: dict[str, str],
    ) -> str:
        """Return the router prompt used for one coalition prompt (debug preview)."""
        del target_tool
        return self._build_router_prompt(prompt, tool_descriptions)

    def _route(
        self,
        prompt: str,
        tool_descriptions: dict[str, str],
    ) -> tuple[str | None, str]:
        """Return the cached or freshly-queried (selected_tool, raw_output) for a prompt."""
        cache_key = (prompt, tuple(sorted(tool_descriptions)))
        if cache_key in self._decision_cache:
            return self._decision_cache[cache_key]

        router_prompt = self._build_router_prompt(prompt, tool_descriptions)
        client = self._get_client()
        result = self._select_tool(client, router_prompt, tool_descriptions)
        self._decision_cache[cache_key] = result
        return result

    def _build_router_prompt(self, prompt: str, tool_descriptions: Mapping[str, str]) -> str:
        """Build a routing-only prompt: no tool_arguments, no assistant_answer."""
        system_prompt, user_request = split_coalition_prompt(prompt)
        tool_lines = "\n".join(
            f"- {tool_name}: {description}" for tool_name, description in tool_descriptions.items()
        )
        return (
            f"{system_prompt}\n\n"
            f"Available tools:\n{tool_lines}\n\n"
            "Select the single best decision for the user request. Return JSON only with this "
            'shape: {"selected_tool":"..."}. Do not include tool arguments or a final answer.\n\n'
            f"Valid selected_tool values: {', '.join(sorted(tool_descriptions))}\n\n"
            f"User request:\n{user_request}"
        )

    def _select_tool(
        self,
        client: object,
        router_prompt: str,
        tool_descriptions: Mapping[str, str],
    ) -> tuple[str | None, str]:
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a tool router. Return JSON only."},
                {"role": "user", "content": router_prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw_output = str(response.choices[0].message.content or "")
        try:
            payload = json.loads(raw_output)
        except json.JSONDecodeError:
            return None, raw_output
        selected_tool = payload.get("selected_tool")
        if not isinstance(selected_tool, str) or selected_tool not in tool_descriptions:
            return None, raw_output
        return selected_tool, raw_output

    def _get_client(self) -> object:
        if self._client is not None:
            return self._client
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            msg = (
                "GROQ_API_KEY is not set. Add it to the environment to use the Groq "
                "deterministic router scorer."
            )
            raise RuntimeError(msg)
        factory = self.client_factory or _default_client_factory
        self._client = factory(api_key)
        return self._client


def _default_client_factory(api_key: str) -> object:
    from groq import Groq

    return Groq(api_key=api_key)


@dataclass
class GroqSoftVoteToolScorer:
    """Groq black-box scorer based on empirical target-tool selection frequency.

    For each coalition prompt, this scorer samples the Groq router ``n_samples``
    times with fixed model, prompt template, temperature, and optional seed. The
    returned soft-vote score is the fraction of sampled router decisions that
    selected ``target_tool``. This is a stochastic behavioral score, not a
    calibrated or contrastive quantity.
    """

    model_name: str = DEFAULT_GROQ_ROUTER_MODEL_ID
    n_samples: int = DEFAULT_GROQ_SOFT_VOTE_N_SAMPLES
    temperature: float = DEFAULT_GROQ_SOFT_VOTE_TEMPERATURE
    max_retries: int = DEFAULT_GROQ_SOFT_VOTE_MAX_RETRIES
    seed: int | None = None
    client_factory: Callable[[str], object] | None = None
    last_debug_outputs: list[dict[str, object]] = field(default_factory=list, init=False)
    _client: object | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n_samples < 1:
            msg = "n_samples must be positive."
            raise ValueError(msg)
        if self.max_retries < 0:
            msg = "max_retries must be non-negative."
            raise ValueError(msg)

    def score_batch(
        self,
        prompts: list[str],
        *,
        target_tool: str,
        tool_descriptions: dict[str, str],
    ) -> list[float]:
        """Return one soft-vote score per coalition prompt."""
        if target_tool not in tool_descriptions:
            msg = f"Target tool {target_tool!r} is not a known decision candidate."
            raise ValueError(msg)

        self.last_debug_outputs = []
        scores = []
        for prompt in prompts:
            score, votes, raw_outputs = self.score_single(
                prompt,
                target_tool=target_tool,
                tool_descriptions=tool_descriptions,
            )
            scores.append(score)
            self.last_debug_outputs.append(
                {
                    "target_tool": target_tool,
                    "score_kind": "soft-vote score",
                    "score_description": "empirical target-tool selection frequency",
                    "selected_tools": votes,
                    "target_matches": [tool_name == target_tool for tool_name in votes],
                    "n_samples": self.n_samples,
                    "temperature": self.temperature,
                    "raw_outputs": raw_outputs,
                    "final_score": score,
                    "prompt_preview": prompt[:240],
                }
            )
        return scores

    def score_single(
        self,
        prompt: str,
        *,
        target_tool: str,
        tool_descriptions: dict[str, str],
    ) -> tuple[float, list[str | None], list[str]]:
        """Return soft-vote score, sampled tool names, and raw router outputs."""
        router_prompt = self._build_router_prompt(prompt, tool_descriptions)
        client = self._get_client()
        votes: list[str | None] = []
        raw_outputs: list[str] = []
        for sample_index in range(self.n_samples):
            selected_tool, raw_output = self._select_tool_with_retries(
                client,
                router_prompt,
                tool_descriptions,
                sample_index=sample_index,
            )
            votes.append(selected_tool)
            raw_outputs.append(raw_output)
        target_count = sum(1 for selected_tool in votes if selected_tool == target_tool)
        return target_count / self.n_samples, votes, raw_outputs

    def build_scoring_prompt(
        self,
        prompt: str,
        *,
        target_tool: str,
        tool_descriptions: dict[str, str],
    ) -> str:
        """Return the router prompt used for one coalition prompt."""
        del target_tool
        return self._build_router_prompt(prompt, tool_descriptions)

    def _build_router_prompt(self, prompt: str, tool_descriptions: Mapping[str, str]) -> str:
        tool_lines = "\n".join(
            f"- {tool_name}: {description}" for tool_name, description in tool_descriptions.items()
        )
        return (
            "You are a tool router.\n"
            "Choose exactly one best tool for the given prompt.\n\n"
            f"Available tools:\n{tool_lines}\n\n"
            f"Prompt:\n{prompt}\n\n"
            "Rules:\n"
            "- Choose exactly one tool.\n"
            "- Judge only based on the visible prompt.\n"
            "- Do not infer missing or removed words unless they are obvious from the visible "
            "text.\n"
            "- If the prompt is incomplete and no tool is clearly needed, choose no_tool.\n"
            "- Return only valid JSON with this schema:\n"
            '{"best_tool": "<tool_name>"}\n\n'
            f"Valid tool names: {', '.join(sorted(tool_descriptions))}"
        )

    def _select_tool_with_retries(
        self,
        client: object,
        router_prompt: str,
        tool_descriptions: Mapping[str, str],
        *,
        sample_index: int,
    ) -> tuple[str | None, str]:
        raw_outputs = []
        for attempt in range(self.max_retries + 1):
            raw_output = self._request_router_output(
                client,
                router_prompt,
                sample_index=sample_index,
                attempt=attempt,
            )
            raw_outputs.append(raw_output)
            selected_tool = self._parse_best_tool(raw_output, tool_descriptions)
            if selected_tool is not None:
                return selected_tool, raw_output
        return None, "\n".join(raw_outputs)

    def _request_router_output(
        self,
        client: object,
        router_prompt: str,
        *,
        sample_index: int,
        attempt: int,
    ) -> str:
        request_kwargs: dict[str, object] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a tool router. Return JSON only."},
                {"role": "user", "content": router_prompt},
            ],
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }
        if self.seed is not None:
            request_kwargs["seed"] = self.seed + sample_index + attempt
        response = client.chat.completions.create(**request_kwargs)
        return str(response.choices[0].message.content or "")

    @staticmethod
    def _parse_best_tool(raw_output: str, tool_descriptions: Mapping[str, str]) -> str | None:
        try:
            payload = json.loads(raw_output)
        except json.JSONDecodeError:
            return None
        best_tool = payload.get("best_tool")
        if not isinstance(best_tool, str) or best_tool not in tool_descriptions:
            return None
        return best_tool

    def _get_client(self) -> object:
        if self._client is not None:
            return self._client
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            msg = (
                "GROQ_API_KEY is not set. Add it to the environment to use the Groq "
                "soft-vote scorer."
            )
            raise RuntimeError(msg)
        factory = self.client_factory or _default_client_factory
        self._client = factory(api_key)
        return self._client
