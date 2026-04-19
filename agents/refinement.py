"""
agents/refinement.py
────────────────────
CSYS  — System Certification    (ARP4754A / ARP4761 completeness)
CSW   — Software Certification  (DO-178C completeness)
"""

from __future__ import annotations

import json
import logging

from agents.base import BaseAgent, SYSTEM_PREAMBLE
from core.model import Task

logger = logging.getLogger(__name__)



# ─────────────────────────────────────────────────────────────────────────────
class CSYSAgent(BaseAgent):
    """
    System Certification Agent.

    Enforces ARP4754A / ARP4761 completeness:
      - §5.1  Aircraft/system requirements must exist
      - §5.2  System requirements must be validated
      - §5.3  System architecture must be defined
      - §6.x  Verification activities must be defined
      - FHA, PSSA, SSA must exist and be linked to requirements
    """

    name = "CSYS"

    REQUIRED_ARTIFACTS = [
        "Aircraft Function List",
        "System Requirements Specification",
        "System Architecture Description",
        "System Verification Plan",
        "FHA Report",
        "PSSA Report",
        "SSA Report",
        "Safety Requirements",
        "System Validation Evidence",
    ]

    def run(self) -> bool:
        self._log("Running system certification compliance pass.")
        produced = self.graph.all_produced_artifacts()
        missing = [a for a in self.REQUIRED_ARTIFACTS if a not in produced]

        if not missing:
            self._log("All required system artifacts present.")
            return False

        self._log("Missing system artifacts: %s", missing)
        ctx = self._rag_context(
            "ARP4754A ARP4761 required artifacts certification compliance"
        )
        system_msg = SYSTEM_PREAMBLE
        user_msg = f"""
You are the CSYS (System Certification) agent.

The following mandatory system artifacts are missing from the process graph:
{missing}

You must add tasks that produce each missing artifact.
Reference the relevant ARP4754A or ARP4761 section for each task.

Current process graph:
{self._graph_json()}
{ctx}

Instructions:
1. Add tasks to produce each missing artifact. Do NOT remove existing tasks.
2. Assign unique IDs higher than all existing IDs.
3. Each new task must reference the relevant standard section.
4. Ensure all task inputs are defined (process inputs or task outputs).
5. Return the complete updated process graph JSON.
""".strip()

        raw = self._call_llm(system_msg, user_msg)
        new_graph = self._parse_graph_response(raw)
        if new_graph is None:
            return False

        existing_ids = {t.id for t in self.graph.tasks}
        changed = False
        for t in new_graph.tasks:
            if t.id not in existing_ids:
                self.graph.tasks.append(t)
                changed = True
        self._log("Changed: %s", changed)
        return changed


# ─────────────────────────────────────────────────────────────────────────────
class CSWAgent(BaseAgent):
    """
    Software Certification Agent.

    Enforces DO-178C completeness:
      - Table A-4: Planning outputs
      - Table A-5: Bidirectional traceability
      - Table A-6: Verification coverage
      - Table A-7: Source code verification
    """

    name = "CSW"

    REQUIRED_ARTIFACTS = [
        # Planning (Table A-1)
        "Plan for Software Aspects of Certification (PSAC)",
        "Software Development Plan (SDP)",
        "Software Verification Plan (SVP)",
        "Software Configuration Management Plan (SCMP)",
        "Software Quality Assurance Plan (SQAP)",
        # Requirements (Table A-2)
        "Software High-Level Requirements",
        "Software Low-Level Requirements",
        # Design (Table A-3)
        "Software Architecture",
        # Implementation (Table A-4)
        "Source Code",
        "Executable Object Code",
        # Verification (Tables A-5 / A-6 / A-7)
        "Software Verification Cases and Procedures",
        "Software Verification Results",
        "Structural Coverage Analysis Report",
        "Traceability Data (HLR to LLR)",
        "Traceability Data (LLR to Source Code)",
        # CM / QA (Tables A-8 / A-9)
        "Software Configuration Index (SCI)",
        "Software Accomplishment Summary (SAS)",
    ]

    def run(self) -> bool:
        self._log("Running software certification compliance pass.")
        produced = self.graph.all_produced_artifacts()
        missing = [a for a in self.REQUIRED_ARTIFACTS if a not in produced]

        if not missing:
            self._log("All required software artifacts present.")
            return False

        self._log("Missing software artifacts: %s", missing)
        ctx = self._rag_context(
            "DO-178C Table A-4 A-5 A-6 A-7 required artifacts verification coverage traceability"
        )
        system_msg = SYSTEM_PREAMBLE
        user_msg = f"""
You are the CSW (Software Certification) agent.

The following mandatory DO-178C software artifacts are missing from the process graph:
{missing}

You must add tasks that produce each missing artifact.

Current process graph:
{self._graph_json()}
{ctx}

Instructions:
1. Add tasks for each missing artifact. Do NOT remove existing tasks.
2. Assign unique IDs higher than all existing IDs.
3. Reference the DO-178C Table or section for each task (e.g. "DO-178C Table A-5").
4. Bidirectional traceability tasks must reference BOTH source and target artifacts.
5. Return the complete updated process graph JSON.
""".strip()

        raw = self._call_llm(system_msg, user_msg)
        new_graph = self._parse_graph_response(raw)
        if new_graph is None:
            return False

        existing_ids = {t.id for t in self.graph.tasks}
        changed = False
        for t in new_graph.tasks:
            if t.id not in existing_ids:
                self.graph.tasks.append(t)
                changed = True
        self._log("Changed: %s", changed)
        return changed
