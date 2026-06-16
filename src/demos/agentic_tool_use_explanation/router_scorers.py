"""Real Groq-router-backed coalition value-function scorer for the agentic demo.

Unlike ``groq_agent.run_groq_tool_inference``, which performs the one real
agent decision (and may execute a demo tool / draft a final answer),
``GroqDeterministicRouterScorer`` is meant to be called once per coalition
during shapiq's coalition sampling. It only asks Groq for a routing decision
and never executes a tool or asks Groq to draft a final answer.
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
