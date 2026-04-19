"""
core/model.py
─────────────
Internal graph representation of the aeronautical development process.

Key design
──────────
  ReferenceRegistry  — REF-xxx IDs ↔ full academic citation strings.
                       Persisted to refs.txt.

  ArtifactRegistry   — ART-xxx IDs ↔ full artifact names.
                       Persisted to artifacts.txt.
                       Tasks store only ART-xxx IDs in Inputs / Outputs.
                       Actual artifact names are resolved at display /
                       verification time, keeping prompts compact.

  Task               — Inputs and Outputs are sets of ART-xxx IDs.
                       AcademicRefs are sets of REF-xxx IDs.

  ProcessGraph       — directed task graph keyed on ART-xxx IDs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Generic ID registry (shared implementation for both artifact and reference)
# ─────────────────────────────────────────────────────────────────────────────

class _IdRegistry:
    """
    Generic bidirectional map: sequential short IDs ↔ full strings.

    Sub-classed by ArtifactRegistry and ReferenceRegistry with different
    ID prefixes and file paths.
    """

    _ID_FORMAT: str   # e.g. "ART-{:03d}" — set by subclass
    _LINE_RE = re.compile(r"^([A-Z]+-\d+)\s*\|\s*(.+)$")

    def __init__(self) -> None:
        self._id_to_value: dict[str, str] = {}
        self._value_to_id: dict[str, str] = {}

    # ── registration ──────────────────────────────────────────────────────────
    def register(self, value: str) -> str:
        """Return the ID for *value*, registering it if new (deduplicates)."""
        value = value.strip()
        if value in self._value_to_id:
            return self._value_to_id[value]
        new_id = self._ID_FORMAT.format(len(self._id_to_value) + 1)
        self._id_to_value[new_id]  = value
        self._value_to_id[value]   = new_id
        return new_id

    def register_many(self, values: list[str]) -> list[str]:
        return [self.register(v) for v in values]

    # ── resolution ────────────────────────────────────────────────────────────
    def resolve(self, id_: str) -> Optional[str]:
        return self._id_to_value.get(id_)

    def resolve_many(self, ids: list[str]) -> list[str]:
        return [self._id_to_value.get(i, i) for i in ids]

    def is_id(self, s: str) -> bool:
        """Return True if *s* is a valid ID for THIS registry (correct prefix)."""
        prefix = self._ID_FORMAT.split("-")[0]   # e.g. "ART" or "REF"
        return bool(re.match(rf"^{prefix}-\d+$", s.strip()))

    def ids(self) -> list[str]:
        return sorted(self._id_to_value.keys())

    def __len__(self) -> int:
        return len(self._id_to_value)

    def __contains__(self, id_: str) -> bool:
        return id_ in self._id_to_value

    def __iter__(self):
        return iter(sorted(self._id_to_value.items()))

    # ── persistence ───────────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        """Write registry to *path* as  ``ID | full value`` lines."""
        lines = [f"{k} | {v}" for k, v in sorted(self._id_to_value.items())]
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

    @classmethod
    def _load_into(cls, instance: "_IdRegistry", path: str) -> None:
        p = Path(path)
        if not p.exists():
            return
        for line in p.read_text(encoding="utf-8").splitlines():
            m = cls._LINE_RE.match(line.strip())
            if m:
                id_, value = m.group(1), m.group(2).strip()
                instance._id_to_value[id_] = value
                instance._value_to_id[value] = id_

    def summary(self) -> str:
        if not self._id_to_value:
            return "(empty)"
        return "\n".join(f"  {k}: {v}" for k, v in sorted(self._id_to_value.items()))

    def to_prompt_table(self) -> str:
        """Compact one-liner table suitable for inclusion in LLM prompts."""
        if not self._id_to_value:
            return "(none)"
        return "\n".join(f"{k}: {v}" for k, v in sorted(self._id_to_value.items()))


# ─────────────────────────────────────────────────────────────────────────────
# ArtifactRegistry
# ─────────────────────────────────────────────────────────────────────────────

class ArtifactRegistry(_IdRegistry):
    """
    Maps ART-xxx IDs ↔ full artifact names.

    Tasks store only ART-xxx IDs in their Inputs and Outputs fields.
    The actual artifact name (e.g. "Software High-Level Requirements") is
    resolved from this registry at display or verification time.

    Design rationale
    ────────────────
    Embedding full artifact names in every task's Inputs/Outputs list inflates
    every LLM prompt.  A short ID like ART-007 conveys the same structural
    information (which task produces what, which task consumes what) at a
    fraction of the token cost.

    Artifact names defined by standards (DO-178C, ARP4754A, ARP4761) must be
    registered using their exact standardised names so that the registry acts
    as the single source of truth for naming.
    """

    _ID_FORMAT = "ART-{:03d}"

    @classmethod
    def load(cls, path: str = "artifacts.txt") -> "ArtifactRegistry":
        reg = cls()
        cls._load_into(reg, path)
        return reg

    def migrate_name(self, name: str) -> str:
        """
        Return ART-xxx for *name*, registering it if not already known.

        Used during graph migration: plain artifact names found in a legacy
        graph are registered on the fly, and the ID is returned for use in
        place of the name.
        """
        return self.register(name)


# ─────────────────────────────────────────────────────────────────────────────
# ReferenceRegistry
# ─────────────────────────────────────────────────────────────────────────────

class ReferenceRegistry(_IdRegistry):
    """
    Maps REF-xxx IDs ↔ full academic citation strings.

    Tasks store only REF-xxx IDs in AcademicRefs.
    Full citations are persisted to refs.txt and never embedded in prompts.
    """

    _ID_FORMAT = "REF-{:03d}"

    @classmethod
    def load(cls, path: str = "refs.txt") -> "ReferenceRegistry":
        reg = cls()
        cls._load_into(reg, path)
        return reg

    def reset(self) -> None:
        """Clear all registered references (called when --reset-acad is set)."""
        self._id_to_value.clear()
        self._value_to_id.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Task
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Task:
    id: int
    parent_id: Optional[int]
    name: str
    description: str
    # Sets of ART-xxx IDs (or plain names in legacy / seed graphs)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    standards: list[str] = field(default_factory=list)
    # Set of REF-xxx IDs (never full citation strings)
    academic_refs: list[str] = field(default_factory=list)
    is_sequential: bool = False

    # ── serialisation ─────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "Id":           self.id,
            "ParentId":     self.parent_id,
            "Name":         self.name,
            "Description":  self.description,
            "Inputs":       sorted(set(self.inputs)),
            "Outputs":      sorted(set(self.outputs)),
            "Standards":    self.standards,
            "AcademicRefs": sorted(set(self.academic_refs)),
            "IsSequential": self.is_sequential,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        return cls(
            id=d["Id"],
            parent_id=d.get("ParentId"),
            name=d["Name"],
            description=d["Description"],
            inputs=list(d.get("Inputs", [])),
            outputs=list(d.get("Outputs", [])),
            standards=d.get("Standards", []),
            academic_refs=list(d.get("AcademicRefs", [])),
            is_sequential=d.get("IsSequential", False),
        )

    # ── artifact ID helpers ───────────────────────────────────────────────────
    def migrate_artifacts(self, art_reg: ArtifactRegistry) -> bool:
        """
        Replace any plain artifact names in inputs/outputs with ART-xxx IDs.

        Returns True if any replacement was made (graph changed).
        Called during graph load to migrate legacy / seed data.
        """
        changed = False
        new_inputs = []
        for a in self.inputs:
            if art_reg.is_id(a):
                new_inputs.append(a)
            else:
                new_inputs.append(art_reg.migrate_name(a))
                changed = True
        new_outputs = []
        for a in self.outputs:
            if art_reg.is_id(a):
                new_outputs.append(a)
            else:
                new_outputs.append(art_reg.migrate_name(a))
                changed = True
        self.inputs  = new_inputs
        self.outputs = new_outputs
        return changed

    # ── decomposition criterion ───────────────────────────────────────────────
    def needs_decomposition(self) -> bool:
        """
        True when the task must be decomposed further:
          1. More than one output artifact (structural).
          2. is_sequential flag set (semantic).
        """
        return self.is_sequential or len(set(self.outputs)) > 1


# ─────────────────────────────────────────────────────────────────────────────
# ProcessGraph
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProcessGraph:
    # Process-level boundary artifacts (ART-xxx IDs or plain names pre-migration)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    tasks: list[Task] = field(default_factory=list)

    # ── id management ─────────────────────────────────────────────────────────
    def _next_id(self) -> int:
        return max((t.id for t in self.tasks), default=0) + 1

    # ── mutation helpers ──────────────────────────────────────────────────────
    def add_task(self, task: Task) -> None:
        if any(t.id == task.id for t in self.tasks):
            raise ValueError(f"Task id {task.id} already exists.")
        self.tasks.append(task)

    def get_task(self, task_id: int) -> Optional[Task]:
        return next((t for t in self.tasks if t.id == task_id), None)

    def replace_task(self, task: Task) -> None:
        self.tasks = [t if t.id != task.id else task for t in self.tasks]

    def children_of(self, task_id: int) -> list[Task]:
        return [t for t in self.tasks if t.parent_id == task_id]

    def roots(self) -> list[Task]:
        return [t for t in self.tasks if t.parent_id is None]

    # ── artifact universe ─────────────────────────────────────────────────────
    def all_produced_artifacts(self) -> set[str]:
        """All ART-xxx IDs available as inputs to downstream tasks."""
        arts: set[str] = set(self.inputs)
        for t in self.tasks:
            arts.update(t.outputs)
        return arts

    def undefined_inputs(self) -> dict[int, list[str]]:
        """Return {task_id: [missing ART-xxx, ...]} for every broken input."""
        produced = self.all_produced_artifacts()
        result: dict[int, list[str]] = {}
        for t in self.tasks:
            missing = [a for a in t.inputs if a not in produced]
            if missing:
                result[t.id] = missing
        return result

    def non_atomic_tasks(self) -> list[Task]:
        return [
            t for t in self.tasks
            if t.needs_decomposition() and not self.children_of(t.id)
        ]

    # ── migration ─────────────────────────────────────────────────────────────
    def migrate_artifacts(self, art_reg: ArtifactRegistry) -> bool:
        """
        Migrate all plain artifact names to ART-xxx IDs throughout the graph.

        Should be called once after loading a seed or legacy graph.
        Process-level inputs and outputs are migrated in addition to tasks.
        Returns True if any changes were made.
        """
        changed = False
        new_inputs = []
        for a in self.inputs:
            if art_reg.is_id(a):
                new_inputs.append(a)
            else:
                new_inputs.append(art_reg.migrate_name(a))
                changed = True
        self.inputs = new_inputs

        new_outputs = []
        for a in self.outputs:
            if art_reg.is_id(a):
                new_outputs.append(a)
            else:
                new_outputs.append(art_reg.migrate_name(a))
                changed = True
        self.outputs = new_outputs

        for t in self.tasks:
            if t.migrate_artifacts(art_reg):
                changed = True

        return changed

    # ── serialisation ─────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "process": {
                "inputs":  sorted(set(self.inputs)),
                "outputs": sorted(set(self.outputs)),
                "tasks":   [t.to_dict() for t in self.tasks],
            }
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "ProcessGraph":
        p = d.get("process", d)
        graph = cls(
            inputs=list(p.get("inputs", [])),
            outputs=list(p.get("outputs", [])),
        )
        for td in p.get("tasks", []):
            graph.tasks.append(Task.from_dict(td))
        return graph

    @classmethod
    def from_json(cls, s: str) -> "ProcessGraph":
        return cls.from_dict(json.loads(s))

    # ── convergence ───────────────────────────────────────────────────────────
    def is_converged(self) -> bool:
        return not self.undefined_inputs() and not self.non_atomic_tasks()

    def convergence_report(self, art_reg: Optional[ArtifactRegistry] = None) -> str:
        """
        Human-readable convergence report.

        If *art_reg* is provided, ART-xxx IDs are resolved to full names.
        """
        def _name(art_id: str) -> str:
            if art_reg:
                return art_reg.resolve(art_id) or art_id
            return art_id

        lines = []
        undef = self.undefined_inputs()
        if undef:
            lines.append("Undefined inputs:")
            for tid, arts in undef.items():
                t = self.get_task(tid)
                resolved = [_name(a) for a in arts]
                lines.append(f"  Task {tid} ({t.name if t else '?'}): {resolved}")

        na = self.non_atomic_tasks()
        if na:
            lines.append("Tasks pending decomposition:")
            for t in na:
                reason = "sequential" if t.is_sequential else "multi-output"
                out_names = [_name(a) for a in t.outputs]
                lines.append(
                    f"  Task {t.id} ({t.name}): reason={reason}, outputs={out_names}"
                )

        return "\n".join(lines) if lines else "Graph converged — all constraints satisfied."
