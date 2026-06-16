"""Real Groq-router-backed coalition value-function scorers for the agentic demo.

Unlike ``groq_agent.run_groq_tool_inference``, which performs the one real
agent decision (and may execute a demo tool / draft a final answer),
these scorers are meant to be called during shapiq's coalition sampling. They
only ask Groq for routing decisions and never execute tools or ask Groq to
draft final answers.
"""

from __future__ import annotations

import ast
import json
import math
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

try:
    from demos.agentic_tool_use_explanation.groq_agent import run_groq_tool_inference
except ModuleNotFoundError:
    from groq_agent import run_groq_tool_inference

try:
    from demos.agentic_tool_use_explanation.scorers import split_coalition_prompt
except ModuleNotFoundError:
    from scorers import split_coalition_prompt

DEFAULT_GROQ_ROUTER_MODEL_ID = "llama-3.1-8b-instant"
DEFAULT_GROQ_SOFT_VOTE_N_SAMPLES = 5
DEFAULT_GROQ_SOFT_VOTE_TEMPERATURE = 0.3
DEFAULT_GROQ_SOFT_VOTE_MAX_RETRIES = 2
DEFAULT_TOOL_MATCH_WEIGHT = 0.5
DEFAULT_ARG_MATCH_WEIGHT = 0.5

# Canonical argument keys follow tool_schemas.EXECUTABLE_TOOL_SCHEMAS. Different
# real agent backends (Groq, Gemini, local HF routers) have been observed to use
# slightly different argument names for the same value -- e.g. groq_agent.py and
# gemini_agent.py both already read either "date_or_time" or "date" for weather_tool.
DEFAULT_ARGUMENT_ALIASES: dict[str, dict[str, str]] = {
    "weather_tool": {
        "date_or_time": "date",
        "time": "date",
        "when": "date",
        "place": "location",
        "city": "location",
    },
    "calculator_tool": {
        "expr": "expression",
        "equation": "expression",
        "calculation": "expression",
    },
    "web_search_tool": {
        "search_query": "query",
        "q": "query",
        "search": "query",
    },
    "no_tool": {},
}


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


@dataclass(frozen=True)
class ToolTrajectory:
    """A real agent decision: the selected tool plus the arguments it called with.

    This is the shape an actual tool-calling agent (Groq/Gemini native function
    calling, a LangChain ``AgentExecutor`` step, or ``hf_router.RouterDecision``)
    produces -- unlike the router-prompt scorers above, which only ever ask for
    a bare ``selected_tool`` string with no arguments.
    """

    selected_tool: str
    tool_arguments: Mapping[str, object] = field(default_factory=dict)


@dataclass
class TrajectoryArgumentMatchScorer:
    """Coalition value function that compares real agent trajectories.

    Unlike the router-prompt scorers in this module, this scorer never asks an
    LLM for a bare routing decision. It assumes ``trajectory_provider`` already
    derives an actual agent trajectory (selected tool + call arguments) for a
    coalition prompt -- e.g. by invoking a real LangChain/Groq tool-calling
    agent on the masked request -- and scores how closely that trajectory
    matches a fixed ``reference_trajectory`` recorded from the full prompt.

    Score for one coalition prompt:
        - ``0.0`` if the coalition's selected tool does not match the
          reference's selected tool.
        - ``1.0`` if the tools match and the reference call has no arguments.
        - ``tool_match_weight + arg_match_weight * argument_match_ratio`` if
          the tools match and the reference call has arguments.
    """

    reference_trajectory: ToolTrajectory
    trajectory_provider: Callable[[str], ToolTrajectory]
    tool_match_weight: float = DEFAULT_TOOL_MATCH_WEIGHT
    arg_match_weight: float = DEFAULT_ARG_MATCH_WEIGHT
    argument_aliases: Mapping[str, Mapping[str, str]] = field(
        default_factory=lambda: DEFAULT_ARGUMENT_ALIASES
    )
    last_debug_outputs: list[dict[str, object]] = field(default_factory=list, init=False)
    _trajectory_cache: dict[str, ToolTrajectory] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.tool_match_weight < 0:
            msg = "tool_match_weight must be non-negative."
            raise ValueError(msg)
        if self.arg_match_weight < 0:
            msg = "arg_match_weight must be non-negative."
            raise ValueError(msg)

    def score_batch(
        self,
        prompts: list[str],
        *,
        target_tool: str,
        tool_descriptions: dict[str, str],
    ) -> list[float]:
        """Return one trajectory-match score per coalition prompt."""
        if target_tool not in tool_descriptions:
            msg = f"Target tool {target_tool!r} is not a known decision candidate."
            raise ValueError(msg)
        if target_tool != self.reference_trajectory.selected_tool:
            msg = (
                f"target_tool {target_tool!r} does not match the reference trajectory's "
                f"selected tool {self.reference_trajectory.selected_tool!r}."
            )
            raise ValueError(msg)

        self.last_debug_outputs = []
        scores = []
        for prompt in prompts:
            coalition_trajectory = self._get_trajectory(prompt)
            score, argument_match_ratio = self._score_trajectory(coalition_trajectory)
            scores.append(score)
            self.last_debug_outputs.append(
                {
                    "target_tool": target_tool,
                    "selected_tool": coalition_trajectory.selected_tool,
                    "coalition_arguments": dict(coalition_trajectory.tool_arguments),
                    "reference_arguments": dict(self.reference_trajectory.tool_arguments),
                    "argument_match_ratio": argument_match_ratio,
                    "final_score": score,
                    "prompt_preview": prompt[:240],
                }
            )
        return scores

    def _get_trajectory(self, prompt: str) -> ToolTrajectory:
        """Return the cached or freshly-provided trajectory for one coalition prompt."""
        if prompt not in self._trajectory_cache:
            self._trajectory_cache[prompt] = self.trajectory_provider(prompt)
        return self._trajectory_cache[prompt]

    def _score_trajectory(
        self,
        coalition_trajectory: ToolTrajectory,
    ) -> tuple[float, float | None]:
        """Return (score, argument_match_ratio) for one coalition trajectory."""
        if coalition_trajectory.selected_tool != self.reference_trajectory.selected_tool:
            return 0.0, None
        reference_arguments = self.reference_trajectory.tool_arguments
        if not reference_arguments:
            return 1.0, None
        argument_match_ratio = self._argument_match_ratio(coalition_trajectory)
        score = self.tool_match_weight + self.arg_match_weight * argument_match_ratio
        return score, argument_match_ratio

    def _argument_match_ratio(self, coalition_trajectory: ToolTrajectory) -> float:
        """Return the fraction of canonical reference argument keys that are matched."""
        tool_name = self.reference_trajectory.selected_tool
        normalized_coalition_arguments = self._normalize_keys(
            tool_name,
            coalition_trajectory.tool_arguments,
        )
        reference_arguments = self.reference_trajectory.tool_arguments
        matched = sum(
            1
            for canonical_key, reference_value in reference_arguments.items()
            if canonical_key in normalized_coalition_arguments
            and self._values_match(
                tool_name,
                reference_value,
                normalized_coalition_arguments[canonical_key],
            )
        )
        return matched / len(reference_arguments)

    def _normalize_keys(
        self,
        tool_name: str,
        arguments: Mapping[str, object],
    ) -> dict[str, object]:
        """Map alias argument keys to their canonical reference key names."""
        alias_map = self.argument_aliases.get(tool_name, {})
        normalized: dict[str, object] = {}
        for key, value in arguments.items():
            canonical_key = alias_map.get(key, key)
            normalized.setdefault(canonical_key, value)
        return normalized

    def _values_match(
        self,
        tool_name: str,
        reference_value: object,
        coalition_value: object,
    ) -> bool:
        """Return whether one reference/coalition argument value pair matches."""
        if tool_name == "calculator_tool":
            numeric_match = _numeric_values_match(reference_value, coalition_value)
            if numeric_match is not None:
                return numeric_match
        normalized_reference = _normalize_text(reference_value)
        normalized_coalition = _normalize_text(coalition_value)
        if not normalized_reference or not normalized_coalition:
            return normalized_reference == normalized_coalition
        return (
            normalized_reference == normalized_coalition
            or normalized_reference in normalized_coalition
            or normalized_coalition in normalized_reference
        )


_PUNCTUATION_PATTERN = re.compile(r"[^\w\s]")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def _normalize_text(value: object) -> str:
    """Lowercase, strip punctuation, and collapse whitespace for fuzzy text matching."""
    text = str(value).strip().lower()
    text = _PUNCTUATION_PATTERN.sub("", text)
    return _WHITESPACE_PATTERN.sub(" ", text).strip()


def _numeric_values_match(reference_value: object, coalition_value: object) -> bool | None:
    """Return arithmetic equality if both values evaluate as numbers, else None."""
    reference_number = _safe_eval_arithmetic(str(reference_value))
    coalition_number = _safe_eval_arithmetic(str(coalition_value))
    if reference_number is None or coalition_number is None:
        return None
    return math.isclose(reference_number, coalition_number, rel_tol=1e-9, abs_tol=1e-9)


def _safe_eval_arithmetic(expression: str) -> float | None:
    """Evaluate a simple arithmetic expression, returning None if it is not numeric."""
    try:
        node = ast.parse(expression, mode="eval")
        return _eval_numeric_node(node.body)
    except Exception:  # noqa: BLE001
        return None


def _eval_numeric_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_numeric_node(node.operand)
    if isinstance(node, ast.BinOp):
        left = _eval_numeric_node(node.left)
        right = _eval_numeric_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
    msg = "unsupported expression"
    raise ValueError(msg)


def build_groq_inference_trajectory_provider(
    model_name: str,
    tool_schemas: Sequence[Mapping[str, object]],
    *,
    tool_context: str,
    client_factory: Callable[[str], object] | None = None,
) -> Callable[[str], ToolTrajectory]:
    """Build a trajectory_provider that re-runs the real Groq agent per coalition.

    Unlike the router-prompt scorers in this module, this calls
    ``groq_agent.run_groq_tool_inference`` -- the same real-agent decision path
    used for the one-shot Inference tab -- on each coalition prompt's masked
    user request, so :class:`TrajectoryArgumentMatchScorer` compares against
    actual tool calls (name + arguments), not bare routing decisions. This
    means one real Groq call per distinct coalition prompt.
    """

    def provider(prompt: str) -> ToolTrajectory:
        system_prompt, user_request = split_coalition_prompt(prompt)
        inference_result = run_groq_tool_inference(
            user_request,
            tool_schemas,
            model_name,
            system_prompt=system_prompt,
            tool_context=tool_context,
            client_factory=client_factory,
        )
        selected_tool = inference_result.selected_tool
        if not isinstance(selected_tool, str) or not selected_tool:
            selected_tool = ""
        return ToolTrajectory(
            selected_tool=selected_tool,
            tool_arguments=dict(inference_result.tool_arguments or {}),
        )

    return provider
