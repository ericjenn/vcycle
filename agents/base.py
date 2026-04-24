"""
agents/base.py
──────────────
Abstract base class shared by all agents.

Every agent:
  • holds a reference to the shared ProcessGraph
  • can query the RAG pipeline for context
  • calls the OpenAI chat model via LangChain
  • logs every LLM exchange (prompt + raw response) to llm_exchanges.log
  • returns a bool from run() indicating whether the graph changed
"""

from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

if TYPE_CHECKING:
    from core.model import ProcessGraph, ReferenceRegistry, ArtifactRegistry
    from rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


# ── dedicated file logger for LLM exchanges ───────────────────────────────────
def _build_exchange_logger(log_path: str = "llm_exchanges.log") -> logging.Logger:
    """
    Return a logger that writes every LLM exchange to *log_path*.

    Each record contains:
        timestamp | agent | direction (SYSTEM / USER / RESPONSE) | content

    The file is opened in append mode so successive runs accumulate.
    A new run header is written when the logger is first created.
    """
    xlogger = logging.getLogger("llm_exchanges")
    if xlogger.handlers:          # already configured (e.g. after module reload)
        return xlogger

    xlogger.setLevel(logging.DEBUG)
    xlogger.propagate = False     # do not bubble up to the root logger

    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    xlogger.addHandler(handler)

    # Write a session separator so different runs are easy to distinguish
    xlogger.info("=" * 80)
    xlogger.info("NEW SESSION  %s", datetime.now().isoformat(sep=" ", timespec="seconds"))
    xlogger.info("=" * 80)
    return xlogger


_exchange_logger = _build_exchange_logger()


def _log_exchange(agent_name: str, system_msg: str, user_msg: str, response: str) -> None:
    """Write one complete LLM exchange to the exchange log file."""
    sep = "-" * 60
    _exchange_logger.debug(
        "\n%s\n[%s]  SYSTEM PROMPT\n%s\n%s\n[%s]  USER PROMPT\n%s\n%s\n[%s]  RESPONSE\n%s\n%s",
        sep, agent_name, sep,
        system_msg,
        sep, agent_name, sep,
        user_msg,
        sep, agent_name, sep,
        response,
        sep,
    )


SYSTEM_PREAMBLE = """
You are an expert in aeronautical system and software development.
You work according to DO-178C, ARP4754A, and ARP4761.

You operate on a structured process graph represented in JSON with the schema:

{
  "process": {
    "inputs":  [<artifact_name>, ...],
    "outputs": [<artifact_name>, ...],
    "tasks": [
      {
        "Id":          <int>,
        "ParentId":    <int|null>,
        "Name":        <str>,
        "Description": <str>,
        "Inputs":      [<artifact_name>, ...],
        "Outputs":     [<artifact_name>, ...],
        "Standards":   [<standard_ref>, ...],
        "AcademicRefs":[<reference>, ...],
        "IsSequential": <bool>
      },
      ...
    ]
  }
}

RULES:
- Return ONLY a valid JSON object matching the schema above.
- Do not wrap the JSON in markdown fences.
- Preserve all existing tasks unless explicitly replacing them.
- Use exact standard artifact names when defined by a standard.
- Every task input must be either a process input or the output of another task.
- Task IDs must be unique positive integers.
- Keep task descriptions concise (≤ 3 sentences) to avoid response truncation.
""".strip()


# ── JSON repair ───────────────────────────────────────────────────────────────
def _repair_truncated_json(raw: str) -> str | None:
    """
    Attempt to salvage a JSON object that was cut short mid-stream.

    Strategy:
      1. Try parsing as-is.
      2. If that fails, find the last complete task entry in the "tasks" array
         (last '}' followed only by whitespace / partial content before EOF),
         truncate there, close the array and the wrapping objects, and retry.

    Returns the repaired JSON string on success, or None if unrecoverable.
    """
    # Pass 1 — already valid
    try:
        json.loads(raw)
        return raw
    except json.JSONDecodeError:
        pass

    # Pass 2 — find the last complete '}' that closes a task object.
    # We walk backwards looking for the last balanced closing brace that is
    # part of the tasks array (not the outer object).
    tasks_start = raw.find('"tasks"')
    if tasks_start == -1:
        return None

    array_open = raw.find('[', tasks_start)
    if array_open == -1:
        return None

    # Find the rightmost position where the JSON is still valid up to a
    # complete task object.  We try progressively shorter suffixes.
    candidate = raw[:raw.rfind('}') + 1] if '}' in raw else raw

    # Close the tasks array, the process object, and the root object
    for suffix in (']}}'  , '\n  ]\n  }\n}', '\n]\n}\n}'):
        attempt = candidate + suffix
        try:
            json.loads(attempt)
            logger.warning(
                "Repaired truncated JSON response (trimmed %d chars, appended '%s').",
                len(raw) - len(candidate), suffix.strip()
            )
            return attempt
        except json.JSONDecodeError:
            continue

    return None



def _parse_retry_after(error_text: str) -> float | None:
    """
    Extract the suggested wait time (in seconds) from an OpenAI rate-limit
    error message.

    The message typically contains phrases like:
      "Please try again in 13.674s."
      "Please try again in 1m30s."
      "try again in 2m5.123s"

    Returns the wait time in seconds as a float, or None if not parseable.
    """
    # Pattern: optional minutes + optional seconds, e.g. "13.674s", "1m30s", "2m5.1s"
    m = re.search(r"try again in (?:(\d+)m)?(\d+(?:\.\d+)?)?s", error_text, re.IGNORECASE)
    if not m:
        return None
    minutes = float(m.group(1)) if m.group(1) else 0.0
    seconds = float(m.group(2)) if m.group(2) else 0.0
    total   = minutes * 60 + seconds
    # Add a small safety margin so we don't hit the limit again immediately
    return total + 1.0 if total > 0 else None

class BaseAgent(ABC):
    """Abstract base for all process-refinement agents."""

    name: str = "BaseAgent"

    def __init__(
        self,
        graph: "ProcessGraph",
        rag: "RAGPipeline",
        registry: "ReferenceRegistry | None" = None,
        art_reg:        "ArtifactRegistry | None"  = None,
        max_refinements: int | None = None,
        model: str = "gpt-4o",
        temperature: float = 0.2,
        max_tokens: int = 16384,
    ):
        self.graph    = graph
        self.rag      = rag
        self.registry = registry    # shared ReferenceRegistry; None → refs disabled
        self.art_reg         = art_reg          # shared ArtifactRegistry
        self.max_refinements = max_refinements  # None = unlimited
        self._refinement_count: int = 0         # refined tasks so far
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # ── public interface ──────────────────────────────────────────────────────
    @abstractmethod
    def run(self) -> bool:
        """
        Execute the agent's pass on self.graph.
        Modifies the graph in-place.
        Returns True if any change was made.
        """

    # ── LLM call with exchange logging and rate-limit retry ─────────────────
    def _call_llm(
        self,
        system_msg: str,
        user_msg: str,
        max_retries: int = 6,
        base_wait: float = 5.0,
    ) -> str:
        """
        Invoke the LLM and return the response text.

        Retries automatically on rate-limit (429) and transient API errors
        using exponential back-off.  The wait time is taken from the error
        message when available ("try again in Xs"), otherwise it doubles from
        *base_wait* on each attempt (5 s, 10 s, 20 s, …).

        Parameters
        ----------
        max_retries : int
            Maximum number of retry attempts after the first failure (default 6).
        base_wait : float
            Initial back-off in seconds when no wait hint is present (default 5).
        """
        from openai import RateLimitError, APIError, APITimeoutError

        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=user_msg),
        ]

        wait = base_wait
        for attempt in range(max_retries + 1):
            try:
                response = self.llm.invoke(messages)
                raw = response.content.strip()
                _log_exchange(self.name, system_msg, user_msg, raw)
                return raw

            except RateLimitError as exc:
                if attempt == max_retries:
                    logger.error(
                        "[%s] Rate limit hit — no retries left. Raising.", self.name
                    )
                    raise

                # Parse suggested wait from the error message if present
                # e.g. "Please try again in 13.674s."
                suggested = _parse_retry_after(str(exc))
                actual_wait = suggested if suggested is not None else wait

                logger.warning(
                    "[%s] Rate limit (attempt %d/%d) — waiting %.1f s before retry.",
                    self.name, attempt + 1, max_retries, actual_wait,
                )
                time.sleep(actual_wait)
                wait *= 2   # double for next attempt if no suggestion is given

            except (APIError, APITimeoutError) as exc:
                if attempt == max_retries:
                    logger.error(
                        "[%s] API error — no retries left: %s", self.name, exc
                    )
                    raise

                logger.warning(
                    "[%s] Transient API error (attempt %d/%d): %s — waiting %.1f s.",
                    self.name, attempt + 1, max_retries, exc, wait,
                )
                time.sleep(wait)
                wait *= 2

        # Should be unreachable
        raise RuntimeError(f"[{self.name}] _call_llm exceeded max_retries")

    # ── graph JSON parsing ────────────────────────────────────────────────────
    def _parse_graph_response(self, raw: str) -> "ProcessGraph | None":
        """
        Parse a JSON process graph from the LLM response.

        Steps:
          1. Strip markdown code fences (``` / ```json).
          2. Extract the first {...} block.
          3. Attempt direct parsing.
          4. On failure, attempt JSON repair (handles truncated responses).
          5. Log the raw response on unrecoverable failure.
        """
        from core.model import ProcessGraph

        # Step 1 — strip markdown fences
        clean = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

        # Step 2 — extract outermost JSON object
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if not match:
            logger.warning("%s: LLM returned no JSON object.", self.name)
            logger.debug("%s: raw response was:\n%s", self.name, raw[:2000])
            return None

        candidate = match.group(0)

        # Step 3 — direct parse
        try:
            return ProcessGraph.from_json(candidate)
        except (json.JSONDecodeError, KeyError):
            pass

        # Step 4 — attempt repair
        repaired = _repair_truncated_json(candidate)
        if repaired is not None:
            try:
                return ProcessGraph.from_json(repaired)
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning("%s: repaired JSON still invalid — %s", self.name, exc)

        # Step 5 — unrecoverable
        logger.warning(
            "%s: could not parse JSON response. "
            "Raw response saved to llm_exchanges.log. "
            "First 500 chars:\n%s",
            self.name, candidate[:500],
        )
        return None

    # ── refinement budget ──────────────────────────────────────────────────────
    def reset_refinement_count(self) -> None:
        """
        Reset the per-iteration refinement counter to zero.

        Called by the orchestrator at the start of each iteration so that
        ``max_refinements`` is interpreted as a *per-iteration* cap (not a
        lifetime cap).  Without this reset, once the budget is exhausted in
        iteration 1 it would remain exhausted for all subsequent iterations,
        preventing any further decomposition regardless of max_iterations.
        """
        self._refinement_count = 0

    @property
    def refinement_budget_exhausted(self) -> bool:
        """
        Return True when this iteration's refinement budget is exhausted.

        ``max_refinements=None`` means unlimited.
        """
        return (
            self.max_refinements is not None
            and self._refinement_count >= self.max_refinements
        )

    def _consume_refinement(self) -> bool:
        """
        Check the per-iteration budget and, if available, consume one slot.

        Returns True if the refinement is allowed, False if the budget is
        exhausted.  Logs a warning when the limit is first hit.
        """
        if self.max_refinements is not None:
            if self._refinement_count >= self.max_refinements:
                self._log(
                    "Refinement budget exhausted (%d/%d) — skipping.",
                    self._refinement_count, self.max_refinements,
                )
                return False
        self._refinement_count += 1
        return True

    # ── RAG context ───────────────────────────────────────────────────────────
    def _rag_context(self, query: str) -> str:
        ctx = self.rag.context_for(query)
        return f"\n\nRelevant standard excerpts:\n{ctx}" if ctx else ""

    # ── graph serialisation helpers ──────────────────────────────────────────
    def _graph_json(self) -> str:
        """
        Return a COMPACT graph summary for use in LLM prompts.

        Structure:
          - Artifact legend (ART-xxx → full name), included only when an
            ArtifactRegistry is available, so the LLM can interpret IDs.
          - Process-level inputs / outputs as ART-xxx IDs.
          - Per task: Id, ParentId, Name, Inputs (ART-xxx), Outputs (ART-xxx).

        Descriptions, standards, and academic refs are omitted — they are
        sent separately via _task_json() when a specific task is being refined.
        """
        compact: dict = {}

        # Artifact legend so the LLM knows what each ART-xxx means
        if self.art_reg and len(self.art_reg) > 0:
            compact["artifact_legend"] = {
                k: v for k, v in self.art_reg
            }

        compact["process"] = {
            "inputs":  sorted(set(self.graph.inputs)),
            "outputs": sorted(set(self.graph.outputs)),
            "tasks": [
                {
                    "Id":       t.id,
                    "ParentId": t.parent_id,
                    "Name":     t.name,
                    "Inputs":   sorted(set(t.inputs)),
                    "Outputs":  sorted(set(t.outputs)),
                }
                for t in self.graph.tasks
            ],
        }
        return json.dumps(compact, indent=2, ensure_ascii=False)

    def _art_legend(self) -> str:
        """
        Return a compact text legend mapping ART-xxx → artifact name,
        suitable for embedding at the top of an LLM prompt.
        """
        if self.art_reg and len(self.art_reg) > 0:
            return "Artifact IDs:\n" + self.art_reg.to_prompt_table()
        return ""

    def _register_artifact(self, name: str) -> str:
        """Register *name* in the ArtifactRegistry and return its ART-xxx ID."""
        if self.art_reg is not None:
            return self.art_reg.register(name)
        return name   # fallback: return plain name if no registry

    def _task_json(self, task) -> str:
        """Return the full JSON for a single task (used in decomposition prompts)."""
        return json.dumps(task.to_dict(), indent=2, ensure_ascii=False)

    def _log(self, msg: str, *args) -> None:
        logger.info("[%s] " + msg, self.name, *args)
