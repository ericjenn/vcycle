"""
agents/consistency.py
─────────────────────
IO    — Consistency Agent  (artifact graph integrity)
ACAD  — Academic Agent     (academic reference enrichment)

Reference handling
──────────────────
ACADAgent asks the LLM for full citation strings, registers each one in the
shared ReferenceRegistry (obtaining a short REF-xxx ID), and stores only the
IDs in task.academic_refs.  Full citations are never embedded in the graph
JSON, keeping prompts compact.

If no registry is available (self.registry is None), ACADAgent stores the
full strings directly as a fallback — this preserves backward compatibility
when the system is used without a registry.
"""

from __future__ import annotations

import json
import logging
import re as _re

from agents.base import BaseAgent, SYSTEM_PREAMBLE

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
class IOAgent(BaseAgent):
    """
    Consistency Agent.

    Validates: for every task input, the artifact must be either:
      - a process-level input, OR
      - an output of another task.

    Resolution strategy:
      1. Fuzzy-match against existing artifact names → remap the input name.
      2. Otherwise → declare as a new process-level input.
      3. Remaining issues → ask the LLM to resolve.
    """

    name = "IO"

    def run(self) -> bool:
        self._log("Running artifact consistency pass.")
        undefined = self.graph.undefined_inputs()
        if not undefined:
            self._log("All artifact inputs are consistent.")
            return False

        self._log("Undefined inputs: %s", undefined)
        changed = False

        produced = self.graph.all_produced_artifacts()
        produced_lower = {a.lower(): a for a in produced}

        for task_id, missing_arts in undefined.items():
            task = self.graph.get_task(task_id)
            if task is None:
                continue
            for art in missing_arts:
                canonical = self._fuzzy_match(art, produced_lower)
                if canonical:
                    task.inputs = [canonical if a == art else a for a in task.inputs]
                    self._log("Task %d: remapped '%s' → '%s'", task_id, art, canonical)
                    changed = True
                else:
                    if art not in self.graph.inputs:
                        self.graph.inputs.append(art)
                        self._log(
                            "Declared new process input: '%s' (required by task %d)",
                            art, task_id,
                        )
                        changed = True

        still_undefined = self.graph.undefined_inputs()
        if still_undefined:
            changed |= self._llm_fix(still_undefined)

        self._log("Changed: %s", changed)
        return changed

    def _fuzzy_match(self, artifact: str, produced_lower: dict[str, str]) -> str | None:
        key = artifact.lower()
        if key in produced_lower:
            return produced_lower[key]
        for k, v in produced_lower.items():
            if key in k or k in key:
                return v
        return None

    def _llm_fix(self, still_undefined: dict[int, list[str]]) -> bool:
        ctx = self._rag_context("artifact consistency traceability process graph")
        user_msg = f"""
You are the IO (Consistency) agent.

The following task inputs are undefined — they are neither process inputs
nor outputs of any other task:

{json.dumps(still_undefined, indent=2)}

Current process graph:
{self._graph_json()}
{ctx}

For each undefined input, you must EITHER:
  (a) Add a new task that produces that artifact, OR
  (b) Declare it as a process-level input if it is an external entry point.

Instructions:
1. Resolve ALL undefined inputs.
2. Do NOT remove existing tasks.
3. Assign unique IDs higher than all existing IDs.
4. Return the complete updated process graph JSON.
""".strip()

        raw = self._call_llm(SYSTEM_PREAMBLE, user_msg)
        new_graph = self._parse_graph_response(raw)
        if new_graph is None:
            return False

        existing_ids = {t.id for t in self.graph.tasks}
        changed = False
        for t in new_graph.tasks:
            if t.id not in existing_ids:
                self.graph.tasks.append(t)
                changed = True
        for inp in new_graph.inputs:
            if inp not in self.graph.inputs:
                self.graph.inputs.append(inp)
                changed = True
        return changed


# ─────────────────────────────────────────────────────────────────────────────
class ACADAgent(BaseAgent):
    """
    Academic Enrichment Agent.

    For each unenriched task, asks the LLM for 1–3 full citation strings.
    Each new citation is registered in the shared ReferenceRegistry, which
    assigns it a unique REF-xxx ID.  Only the IDs are written into
    task.academic_refs — full strings are never stored in the graph.

    If self.registry is None (no registry configured), full citation strings
    are stored directly as a backward-compatible fallback.
    """

    name = "ACAD"

    _SKIP_KEYWORDS = {"plan", "configuration", "baseline", "release"}

    def run(self) -> bool:
        self._log("Running academic enrichment pass.")
        changed = False

        for task in self.graph.tasks:
            if task.academic_refs:
                continue  # already enriched
            if any(kw in task.name.lower() for kw in self._SKIP_KEYWORDS):
                continue

            refs = self._fetch_and_register(task)
            if refs:
                task.academic_refs = refs
                changed = True

        self._log(
            "Changed: %s  |  Registry size: %d",
            changed,
            len(self.registry) if self.registry else 0,
        )
        return changed

    def _fetch_and_register(self, task) -> list[str]:
        """
        Ask the LLM for citation strings, register them, return REF-xxx IDs.

        The LLM always returns full citation strings — registration and ID
        assignment happen locally so the LLM prompt never needs to know
        what IDs already exist.
        """
        query = (
            f"machine learning automation {task.name} "
            f"aerospace software verification"
        )
        ctx = self._rag_context(query)

        # Send a minimal task summary — exclude AcademicRefs (already empty)
        # and IsSequential to keep the prompt concise.
        task_summary = {
            "Id": task.id,
            "Name": task.name,
            "Description": task.description,
            "Standards": task.standards,
        }

        user_msg = f"""
You are the ACAD (Academic) agent.

For the following aeronautical development task, provide 1–3 academic
references (papers, books, or standards) relevant to automation or
machine-learning support for this activity. Cite published, citable works.

Task:
{json.dumps(task_summary, indent=2)}
{ctx if ctx else ""}

Return ONLY a JSON array of full citation strings:
["Author et al. (Year). Title. Venue/Publisher.", ...]

If no relevant academic work exists, return: []
""".strip()

        raw = self._call_llm(SYSTEM_PREAMBLE, user_msg)

        # Parse the citation strings returned by the LLM
        clean = _re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
        match = _re.search(r"\[.*\]", clean, _re.DOTALL)
        if not match:
            return []
        try:
            citations = json.loads(match.group(0))
            citations = [c for c in citations if isinstance(c, str) and c.strip()]
        except json.JSONDecodeError:
            return []

        if not citations:
            return []

        # Register citations and return IDs (or raw strings if no registry)
        if self.registry is not None:
            ids = self.registry.register_many(citations)
            self._log(
                "Task %d (%s): registered %d refs → %s",
                task.id, task.name, len(ids), ids,
            )
            return ids
        else:
            # Fallback: store full strings (backward-compatible)
            return citations
