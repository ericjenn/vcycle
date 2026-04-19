"""
agents/engineering.py
─────────────────────
ESYS  — System Engineering Agent   (ARP4754A)
ESAF  — Safety Assessment Agent    (ARP4761)
ESW   — Software Engineering Agent (DO-178C)

Each agent operates in two complementary modes per iteration:

  extend  — add missing tasks that fall within the agent's domain
  refine  — inspect existing leaf tasks owned by the agent and decompose
             any that represent a sequence of distinct intellectual or
             manual activities, replacing them with atomic child tasks

Both modes run in every iteration.  The loop stops when neither mode
produces any change across all agents.
"""

from __future__ import annotations

import json
import logging
import re

from agents.base import BaseAgent, SYSTEM_PREAMBLE
from core.model import Task

logger = logging.getLogger(__name__)


# ── shared helpers ─────────────────────────────────────────────────────────────
def _merge_graph(agent: BaseAgent, new_graph) -> bool:
    """
    Merge tasks/inputs/outputs from *new_graph* into *agent.graph*.

    - New task IDs are appended.
    - Existing tasks are replaced if outputs or description changed
      (allows refinement to update a coarse task in-place).
    - Process-level inputs/outputs are union-merged.
    """
    existing_ids = {t.id for t in agent.graph.tasks}
    changed = False

    for t in new_graph.tasks:
        if t.id not in existing_ids:
            agent.graph.tasks.append(t)
            changed = True
        else:
            existing = agent.graph.get_task(t.id)
            if existing and (
                set(t.outputs) != set(existing.outputs)
                or t.description != existing.description
                or t.is_sequential != existing.is_sequential
            ):
                agent.graph.replace_task(t)
                changed = True

    for inp in new_graph.inputs:
        if inp not in agent.graph.inputs:
            agent.graph.inputs.append(inp)
            changed = True

    for out in new_graph.outputs:
        if out not in agent.graph.outputs:
            agent.graph.outputs.append(out)
            changed = True

    return changed


def _parse_subtask_array(agent: BaseAgent, raw: str, parent_id: int) -> list[Task]:
    """Parse a JSON array of subtask dicts; enforce parent_id; skip duplicate IDs."""
    clean = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    match = re.search(r"\[.*\]", clean, re.DOTALL)
    if not match:
        logger.warning("[%s] No JSON array in decomposition response.", agent.name)
        return []
    try:
        dicts = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        logger.warning("[%s] JSON parse error: %s", agent.name, exc)
        return []

    tasks = []
    existing_ids = {t.id for t in agent.graph.tasks}
    for d in dicts:
        if d.get("Id") in existing_ids:
            continue
        try:
            child = Task.from_dict(d)
            child.parent_id = parent_id
            tasks.append(child)
        except Exception as exc:
            logger.warning("[%s] Could not parse subtask: %s", agent.name, exc)
    return tasks


_DECOMPOSE_FOOTER = """
If the task IS atomic, return an empty array: []

If sequential, return child tasks as a JSON array:
[
  {{
    "Id": <int, higher than all existing IDs in the graph>,
    "ParentId": {parent_id},
    "Name": "<single activity name>",
    "Description": "<precise, verifiable, single-activity description>",
    "Inputs": [...],
    "Outputs": ["<single primary output>"],
    "Standards": ["{standard_ref}"],
    "AcademicRefs": [],
    "IsSequential": false
  }},
  ...
]
""".strip()

_ATOMICITY_DEFINITION = """
An atomic task:
  - Represents ONE single intellectual or manual activity
  - Produces at most one primary output artifact
  - Can be verified by a single method (review, analysis, or test)
  - Cannot be described as "do A, then do B"
""".strip()


# ── ESYS ──────────────────────────────────────────────────────────────────────
class ESYSAgent(BaseAgent):
    """System Engineering Agent (ARP4754A)."""

    name = "ESYS"

    _DOMAIN_KEYWORDS = {
        "aircraft", "system", "function", "allocation", "architecture",
        "interface", "arp4754", "arp 4754", "safety requirement",
        "functional hazard", "pssa", "ssa", "fha",
    }

    def run(self) -> bool:
        self._log("Running (extend + refine).")
        return self._extend() | self._refine()

    # ── extend ────────────────────────────────────────────────────────────────
    def _extend(self) -> bool:
        ctx = self._rag_context(
            "ARP4754A system development process tasks artifacts "
            "requirements architecture allocation verification"
        )
        user_msg = f"""
You are the ESYS (System Engineering) agent — EXTEND mode.

Add any MISSING system-level tasks to the graph following ARP4754A. Cover:
  - Aircraft / system function identification
  - System Requirements definition and allocation
  - System Architecture definition
  - System-to-software requirements allocation
  - System-level verification planning (§6)
  - Interface definition

Current process graph:
{self._graph_json()}
{ctx}

Instructions:
1. Add ONLY missing tasks. Do NOT remove or modify existing tasks.
2. Assign unique IDs higher than all existing IDs.
3. Each task must reference at least one ARP4754A section.
4. Every input must be a process input or an output of another task.
5. Return the complete updated process graph JSON.
""".strip()

        raw = self._call_llm(SYSTEM_PREAMBLE, user_msg)
        ng = self._parse_graph_response(raw)
        if ng is None:
            self._log("extend: no valid graph returned.")
            return False
        changed = _merge_graph(self, ng)
        self._log("extend: changed=%s", changed)
        return changed

    # ── refine ────────────────────────────────────────────────────────────────
    def _refine(self) -> bool:
        candidates = [
            t for t in self.graph.tasks
            if not self.graph.children_of(t.id) and self._owns(t)
        ]
        if not candidates:
            self._log("refine: no domain leaf tasks.")
            return False
        changed = False
        for task in candidates:
            if self._decompose(task):
                changed = True
        return changed

    def _owns(self, task: Task) -> bool:
        text = (task.name + " " + task.description).lower()
        return any(kw in text for kw in self._DOMAIN_KEYWORDS)

    def _decompose(self, task: Task) -> bool:
        ctx = self._rag_context(f"ARP4754A decompose {task.name}")
        footer = _DECOMPOSE_FOOTER.format(parent_id=task.id, standard_ref="ARP4754A §x.x")
        user_msg = f"""
You are the ESYS (System Engineering) agent — REFINE mode.

Examine the task below. Decide if it describes a SEQUENCE of distinct
intellectual or manual activities. If so, decompose it into atomic children.

{_ATOMICITY_DEFINITION}

Task to examine:
{json.dumps(task.to_dict(), indent=2)}

Current process graph (for context / ID allocation):
{self._graph_json()}
{ctx}

{footer}
""".strip()

        raw = self._call_llm(SYSTEM_PREAMBLE, user_msg)
        subtasks = _parse_subtask_array(self, raw, task.id)
        if not subtasks:
            return False
        for child in subtasks:
            self.graph.add_task(child)
        task.is_sequential = True
        self._log("refine: decomposed task %d (%s) → %d children.",
                  task.id, task.name, len(subtasks))
        return True


# ── ESAF ──────────────────────────────────────────────────────────────────────
class ESAFAgent(BaseAgent):
    """Safety Assessment Agent (ARP4761)."""

    name = "ESAF"

    _DOMAIN_KEYWORDS = {
        "safety", "hazard", "fha", "pssa", "ssa", "fta", "fmea", "cma",
        "failure", "fault", "arp4761", "arp 4761", "dal", "design assurance",
    }

    def run(self) -> bool:
        self._log("Running (extend + refine).")
        return self._extend() | self._refine()

    def _extend(self) -> bool:
        ctx = self._rag_context(
            "ARP4761 FHA PSSA SSA FTA FMEA safety assessment tasks artifacts"
        )
        user_msg = f"""
You are the ESAF (Safety Assessment) agent — EXTEND mode.

Add any MISSING safety assessment tasks following ARP4761. Cover:
  - Functional Hazard Assessment (FHA)
  - Preliminary System Safety Assessment (PSSA)
  - System Safety Assessment (SSA)
  - Fault Tree Analysis (FTA)
  - Failure Modes and Effects Analysis (FMEA)
  - Common Mode Analysis (CMA)
  - Safety requirements derivation and DAL assignment

Each safety task must be linked to system requirements and verification activities.

Current process graph:
{self._graph_json()}
{ctx}

Instructions:
1. Add ONLY missing tasks. Do NOT remove or modify existing tasks.
2. Assign unique IDs higher than all existing IDs.
3. Each task must reference the relevant ARP4761 section.
4. FHA Report, PSSA Report, SSA Report must appear as task outputs.
5. Return the complete updated process graph JSON.
""".strip()

        raw = self._call_llm(SYSTEM_PREAMBLE, user_msg)
        ng = self._parse_graph_response(raw)
        if ng is None:
            self._log("extend: no valid graph returned.")
            return False
        changed = _merge_graph(self, ng)
        self._log("extend: changed=%s", changed)
        return changed

    def _refine(self) -> bool:
        candidates = [
            t for t in self.graph.tasks
            if not self.graph.children_of(t.id) and self._owns(t)
        ]
        if not candidates:
            self._log("refine: no domain leaf tasks.")
            return False
        changed = False
        for task in candidates:
            if self._decompose(task):
                changed = True
        return changed

    def _owns(self, task: Task) -> bool:
        text = (task.name + " " + task.description).lower()
        return any(kw in text for kw in self._DOMAIN_KEYWORDS)

    def _decompose(self, task: Task) -> bool:
        ctx = self._rag_context(f"ARP4761 safety task decompose {task.name}")
        footer = _DECOMPOSE_FOOTER.format(parent_id=task.id, standard_ref="ARP4761 §x.x")
        user_msg = f"""
You are the ESAF (Safety Assessment) agent — REFINE mode.

Examine the safety task below. Decide if it describes a SEQUENCE of distinct
activities. If so, decompose it into atomic children.

{_ATOMICITY_DEFINITION}

Task to examine:
{json.dumps(task.to_dict(), indent=2)}

Current process graph (for context / ID allocation):
{self._graph_json()}
{ctx}

{footer}
""".strip()

        raw = self._call_llm(SYSTEM_PREAMBLE, user_msg)
        subtasks = _parse_subtask_array(self, raw, task.id)
        if not subtasks:
            return False
        for child in subtasks:
            self.graph.add_task(child)
        task.is_sequential = True
        self._log("refine: decomposed task %d (%s) → %d children.",
                  task.id, task.name, len(subtasks))
        return True


# ── ESW ───────────────────────────────────────────────────────────────────────
class ESWAgent(BaseAgent):
    """Software Engineering Agent (DO-178C)."""

    name = "ESW"

    _DOMAIN_KEYWORDS = {
        "software", "do-178", "do 178", "hlr", "llr", "source code",
        "verification", "testing", "coverage", "psac", "sdp", "svp",
        "scmp", "sqap", "configuration management", "quality assurance",
        "executable", "coding", "unit test", "integration test, debug, wcet analsysis"
    }

    def run(self) -> bool:
        self._log("Running (extend + refine).")
        return self._extend() | self._refine()

    def _extend(self) -> bool:
        ctx = self._rag_context(
            "DO-178C software lifecycle planning requirements design coding "
            "verification configuration management quality assurance"
        )
        user_msg = f"""
You are the ESW (Software Engineering) agent — EXTEND mode.

Add any MISSING software lifecycle tasks following DO-178C. Cover:
  - Software Planning (PSAC, SDP, SVP, SCMP, SQAP)
  - High-Level Requirements (HLR) development
  - Low-Level Requirements (LLR) development
  - Software Architecture
  - Source Code implementation
  - Debugging
  - WCET analysis
  - Optimization (memory, speed)
  - Software Verification (unit tests, integration tests, coverage analysis)
  - Configuration Management
  - Quality Assurance

Current process graph:
{self._graph_json()}
{ctx}

Instructions:
1. Add ONLY missing tasks. Do NOT remove or modify existing tasks.
2. Assign unique IDs higher than all existing IDs.
3. Each task must reference at least one DO-178C Table A-x or section.
4. Use exact artifact names as defined by DO-178C.
5. Every input must be a process input or an output of another task.
6. Return the complete updated process graph JSON.
""".strip()

        raw = self._call_llm(SYSTEM_PREAMBLE, user_msg)
        ng = self._parse_graph_response(raw)
        if ng is None:
            self._log("extend: no valid graph returned.")
            return False
        changed = _merge_graph(self, ng)
        self._log("extend: changed=%s", changed)
        return changed

    def _refine(self) -> bool:
        candidates = [
            t for t in self.graph.tasks
            if not self.graph.children_of(t.id) and self._owns(t)
        ]
        if not candidates:
            self._log("refine: no domain leaf tasks.")
            return False
        changed = False
        for task in candidates:
            if self._decompose(task):
                changed = True
        return changed

    def _owns(self, task: Task) -> bool:
        text = (task.name + " " + task.description).lower()
        return any(kw in text for kw in self._DOMAIN_KEYWORDS)

    def _decompose(self, task: Task) -> bool:
        ctx = self._rag_context(f"DO-178C software task decompose {task.name}")
        footer = _DECOMPOSE_FOOTER.format(parent_id=task.id, standard_ref="DO-178C §x.x")
        user_msg = f"""
You are the ESW (Software Engineering) agent — REFINE mode.

Examine the software task below. Decide if it describes a SEQUENCE of distinct
intellectual or manual activities. If so, decompose it into atomic children.

{_ATOMICITY_DEFINITION}

Task to examine:
{json.dumps(task.to_dict(), indent=2)}

Current process graph (for context / ID allocation):
{self._graph_json()}
{ctx}

{footer}
""".strip()

        raw = self._call_llm(SYSTEM_PREAMBLE, user_msg)
        subtasks = _parse_subtask_array(self, raw, task.id)
        if not subtasks:
            return False
        for child in subtasks:
            self.graph.add_task(child)
        task.is_sequential = True
        self._log("refine: decomposed task %d (%s) → %d children.",
                  task.id, task.name, len(subtasks))
        return True
