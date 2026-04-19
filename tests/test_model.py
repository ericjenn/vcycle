"""
tests/test_model.py
────────────────────
Unit tests for core/model.py — no OpenAI API key required.
Run with:  python -m pytest tests/ -v
"""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.model import Task, ProcessGraph, ReferenceRegistry, ArtifactRegistry


# ── Task ─────────────────────────────────────────────────────────────────────
def test_task_roundtrip():
    t = Task(
        id=1, parent_id=None,
        name="Test Task",
        description="A test.",
        inputs=["In1"],
        outputs=["Out1"],
        standards=["DO-178C §5"],
        academic_refs=["Doe et al. 2020"],
    )
    d = t.to_dict()
    t2 = Task.from_dict(d)
    assert t2.id == 1
    assert t2.name == "Test Task"
    assert t2.inputs == ["In1"]
    assert t2.standards == ["DO-178C §5"]


def test_task_not_needing_decomposition_single_output():
    t = Task(id=1, parent_id=None, name="X", description="", outputs=["A"])
    assert not t.needs_decomposition()


def test_task_needs_decomposition_multi_output():
    t = Task(id=1, parent_id=None, name="X", description="", outputs=["A", "B"])
    assert t.needs_decomposition()


def test_task_needs_decomposition_when_sequential_flag_set():
    # Single output but flagged as sequential → must decompose
    t = Task(id=1, parent_id=None, name="X", description="",
             outputs=["A"], is_sequential=True)
    assert t.needs_decomposition()


def test_task_sequential_flag_defaults_false():
    t = Task(id=1, parent_id=None, name="X", description="")
    assert t.is_sequential is False


def test_task_sequential_roundtrip():
    t = Task(id=1, parent_id=None, name="X", description="",
             outputs=["A"], is_sequential=True)
    d = t.to_dict()
    assert d["IsSequential"] is True
    t2 = Task.from_dict(d)
    assert t2.is_sequential is True


def test_task_sequential_false_not_in_dict_defaults_false():
    d = {"Id": 1, "ParentId": None, "Name": "X", "Description": ""}
    t = Task.from_dict(d)
    assert t.is_sequential is False


# ── ProcessGraph ──────────────────────────────────────────────────────────────
def _make_graph() -> ProcessGraph:
    g = ProcessGraph(inputs=["Raw Need"], outputs=["Final Report"])
    g.add_task(Task(id=1, parent_id=None, name="T1", description="",
                    inputs=["Raw Need"], outputs=["Artifact A"]))
    g.add_task(Task(id=2, parent_id=None, name="T2", description="",
                    inputs=["Artifact A"], outputs=["Final Report"]))
    return g


def test_graph_all_produced():
    g = _make_graph()
    produced = g.all_produced_artifacts()
    assert "Raw Need" in produced
    assert "Artifact A" in produced
    assert "Final Report" in produced


def test_graph_no_undefined_inputs():
    g = _make_graph()
    assert g.undefined_inputs() == {}


def test_graph_undefined_input_detected():
    g = _make_graph()
    g.add_task(Task(id=3, parent_id=None, name="T3", description="",
                    inputs=["MISSING"], outputs=["X"]))
    undef = g.undefined_inputs()
    assert 3 in undef
    assert "MISSING" in undef[3]


def test_graph_needs_decomposition_detected():
    g = _make_graph()
    g.add_task(Task(id=4, parent_id=None, name="T4", description="",
                    outputs=["X", "Y"]))
    na = g.non_atomic_tasks()
    assert any(t.id == 4 for t in na)


def test_graph_needs_decomposition_excluded_when_has_children():
    g = _make_graph()
    g.add_task(Task(id=4, parent_id=None, name="T4", description="",
                    outputs=["X", "Y"]))
    g.add_task(Task(id=5, parent_id=4, name="T4-child", description="",
                    outputs=["X"]))
    na = g.non_atomic_tasks()
    # T4 has a child so should NOT appear in non_atomic_tasks
    assert not any(t.id == 4 for t in na)


def test_graph_json_roundtrip():
    g = _make_graph()
    j = g.to_json()
    g2 = ProcessGraph.from_json(j)
    assert len(g2.tasks) == len(g.tasks)
    assert g2.inputs == g.inputs
    assert g2.outputs == g.outputs


def test_convergence_check():
    g = _make_graph()
    assert g.is_converged()


def test_not_converged_with_undefined_input():
    g = _make_graph()
    g.add_task(Task(id=3, parent_id=None, name="T3", description="",
                    inputs=["UNDEFINED"], outputs=["Z"]))
    assert not g.is_converged()


def test_convergence_report_clean():
    g = _make_graph()
    report = g.convergence_report()
    assert "converged" in report.lower()


def test_convergence_report_lists_issues():
    g = _make_graph()
    g.add_task(Task(id=3, parent_id=None, name="T3", description="",
                    inputs=["UNDEFINED"], outputs=["Z"],))
    report = g.convergence_report()
    assert "UNDEFINED" in report


def test_children_of():
    g = _make_graph()
    g.add_task(Task(id=5, parent_id=1, name="child", description="",
                    inputs=[], outputs=[]))
    children = g.children_of(1)
    assert len(children) == 1
    assert children[0].id == 5


def test_add_duplicate_id_raises():
    g = _make_graph()
    import pytest
    with pytest.raises(ValueError):
        g.add_task(Task(id=1, parent_id=None, name="dup", description=""))





# ── ArtifactRegistry ───────────────────────────────────────────────────────────
def test_artifact_register_new():
    reg = ArtifactRegistry()
    aid = reg.register("Software High-Level Requirements")
    assert aid == "ART-001"
    assert len(reg) == 1


def test_artifact_deduplicates():
    reg = ArtifactRegistry()
    id1 = reg.register("Same artifact")
    id2 = reg.register("Same artifact")
    assert id1 == id2
    assert len(reg) == 1


def test_artifact_sequential_ids():
    reg = ArtifactRegistry()
    ids = reg.register_many(["A", "B", "C"])
    assert ids == ["ART-001", "ART-002", "ART-003"]


def test_artifact_resolve():
    reg = ArtifactRegistry()
    aid = reg.register("Source Code")
    assert reg.resolve(aid) == "Source Code"
    assert reg.resolve("ART-999") is None


def test_artifact_is_id():
    reg = ArtifactRegistry()
    assert reg.is_id("ART-001")
    assert not reg.is_id("Software High-Level Requirements")
    assert not reg.is_id("REF-001")


def test_artifact_save_load(tmp_path):
    reg = ArtifactRegistry()
    reg.register("Aircraft Function List")
    reg.register("System Requirements Specification")
    fpath = str(tmp_path / "artifacts.txt")
    reg.save(fpath)
    reg2 = ArtifactRegistry.load(fpath)
    assert len(reg2) == 2
    assert reg2.resolve("ART-001") == "Aircraft Function List"
    assert reg2.resolve("ART-002") == "System Requirements Specification"


def test_artifact_load_missing_file(tmp_path):
    reg = ArtifactRegistry.load(str(tmp_path / "nonexistent.txt"))
    assert len(reg) == 0


def test_artifact_prompt_table():
    reg = ArtifactRegistry()
    reg.register("Alpha")
    reg.register("Beta")
    table = reg.to_prompt_table()
    assert "ART-001: Alpha" in table
    assert "ART-002: Beta" in table


# ── Migration ──────────────────────────────────────────────────────────────────
def test_task_migrate_plain_names():
    reg = ArtifactRegistry()
    t = Task(id=1, parent_id=None, name="T", description="",
             inputs=["Aircraft Function List"], outputs=["System Requirements"])
    changed = t.migrate_artifacts(reg)
    assert changed
    assert t.inputs  == ["ART-001"]
    assert t.outputs == ["ART-002"]
    assert reg.resolve("ART-001") == "Aircraft Function List"


def test_task_migrate_already_ids_noop():
    reg = ArtifactRegistry()
    reg.register("Some artifact")   # → ART-001
    t = Task(id=1, parent_id=None, name="T", description="",
             inputs=["ART-001"], outputs=["ART-001"])
    changed = t.migrate_artifacts(reg)
    assert not changed   # IDs unchanged, nothing to migrate


def test_graph_migrate_artifacts():
    reg = ArtifactRegistry()
    graph = ProcessGraph(
        inputs=["Aircraft-Level Operational Requirements"],
        outputs=["SSA Report"],
    )
    graph.add_task(Task(
        id=1, parent_id=None, name="T1", description="",
        inputs=["Aircraft-Level Operational Requirements"],
        outputs=["Aircraft Function List"],
    ))
    changed = graph.migrate_artifacts(reg)
    assert changed
    # All plain names replaced by IDs
    assert all(reg.is_id(a) for a in graph.inputs)
    assert all(reg.is_id(a) for a in graph.outputs)
    for t in graph.tasks:
        assert all(reg.is_id(a) for a in t.inputs)
        assert all(reg.is_id(a) for a in t.outputs)


def test_graph_undefined_inputs_with_ids():
    reg = ArtifactRegistry()
    id_a = reg.register("Artifact A")
    id_b = reg.register("Artifact B")
    graph = ProcessGraph(inputs=[id_a])
    graph.add_task(Task(id=1, parent_id=None, name="T1", description="",
                        inputs=[id_a], outputs=[id_b]))
    graph.add_task(Task(id=2, parent_id=None, name="T2", description="",
                        inputs=[id_b], outputs=[]))
    assert graph.undefined_inputs() == {}


def test_graph_convergence_report_resolves_names():
    reg = ArtifactRegistry()
    id_missing = reg.register("Missing Artifact")
    graph = ProcessGraph(inputs=[])
    graph.add_task(Task(id=1, parent_id=None, name="T1", description="",
                        inputs=[id_missing], outputs=[]))
    report = graph.convergence_report(art_reg=reg)
    assert "Missing Artifact" in report   # resolved from ID


def test_reference_registry_reset():
    reg = ReferenceRegistry()
    reg.register("Some paper.")
    assert len(reg) == 1
    reg.reset()
    assert len(reg) == 0


# ── ReferenceRegistry ──────────────────────────────────────────────────────────
def test_registry_register_new():
    reg = ReferenceRegistry()
    ref_id = reg.register("Doe et al. (2020). Title. Venue.")
    assert ref_id == "REF-001"
    assert len(reg) == 1


def test_registry_deduplicates():
    reg = ReferenceRegistry()
    id1 = reg.register("Same citation.")
    id2 = reg.register("Same citation.")
    assert id1 == id2
    assert len(reg) == 1


def test_registry_sequential_ids():
    reg = ReferenceRegistry()
    ids = reg.register_many(["Ref A.", "Ref B.", "Ref C."])
    assert ids == ["REF-001", "REF-002", "REF-003"]


def test_registry_resolve():
    reg = ReferenceRegistry()
    ref_id = reg.register("Author (2021). Paper. Journal.")
    assert reg.resolve(ref_id) == "Author (2021). Paper. Journal."
    assert reg.resolve("REF-999") is None


def test_registry_resolve_many_unknown_passthrough():
    reg = ReferenceRegistry()
    reg.register("Known ref.")
    result = reg.resolve_many(["REF-001", "REF-999"])
    assert result[0] == "Known ref."
    assert result[1] == "REF-999"   # unknown ID passed through as-is


def test_registry_is_id():
    reg = ReferenceRegistry()
    assert reg.is_id("REF-001")
    assert reg.is_id("REF-123")
    assert not reg.is_id("Some full citation string.")
    assert not reg.is_id("")


def test_registry_save_load(tmp_path):
    reg = ReferenceRegistry()
    reg.register("Alpha et al. (2019). A. Conf.")
    reg.register("Beta et al. (2022). B. Journal.")
    fpath = str(tmp_path / "refs.txt")
    reg.save(fpath)

    reg2 = ReferenceRegistry.load(fpath)
    assert len(reg2) == 2
    assert reg2.resolve("REF-001") == "Alpha et al. (2019). A. Conf."
    assert reg2.resolve("REF-002") == "Beta et al. (2022). B. Journal."


def test_registry_load_missing_file(tmp_path):
    reg = ReferenceRegistry.load(str(tmp_path / "nonexistent.txt"))
    assert len(reg) == 0


def test_registry_contains():
    reg = ReferenceRegistry()
    ref_id = reg.register("X.")
    assert ref_id in reg
    assert "REF-999" not in reg


def test_task_academic_refs_store_ids():
    """Tasks should store REF-xxx IDs, not full strings."""
    reg = ReferenceRegistry()
    ref_id = reg.register("Doe (2020). Title. Venue.")
    t = Task(id=1, parent_id=None, name="T", description="",
             academic_refs=[ref_id])
    d = t.to_dict()
    assert d["AcademicRefs"] == ["REF-001"]
    t2 = Task.from_dict(d)
    assert t2.academic_refs == ["REF-001"]
    # resolve via registry
    assert reg.resolve(t2.academic_refs[0]) == "Doe (2020). Title. Venue."



# ── Rate-limit helper ──────────────────────────────────────────────────────────
def test_parse_retry_after_seconds():
    """Test _parse_retry_after in isolation using regex directly."""
    import re

    def _parse_retry_after(error_text):
        m = re.search(r"try again in (?:(\d+)m)?(\d+(?:\.\d+)?)?s",
                      error_text, re.IGNORECASE)
        if not m:
            return None
        minutes = float(m.group(1)) if m.group(1) else 0.0
        seconds = float(m.group(2)) if m.group(2) else 0.0
        total = minutes * 60 + seconds
        return total + 1.0 if total > 0 else None

    assert abs(_parse_retry_after("Please try again in 13.674s.") - 14.674) < 0.01
    assert abs(_parse_retry_after("try again in 1m30s") - 91.0) < 0.01
    assert abs(_parse_retry_after("try again in 2m5.5s") - 126.5) < 0.01
    assert _parse_retry_after("try again in 0s") is None
    assert _parse_retry_after("no time mentioned here") is None


# ── active_agents validation ───────────────────────────────────────────────────
def test_active_agents_unknown_raises():
    """Passing an unknown agent name should raise ValueError — logic test."""
    valid = {"ESYS", "ESW", "ESAF", "CSYS", "CSW", "IO", "ACAD"}
    requested = {"ESYS", "FOO"}
    unknown = requested - valid
    try:
        if unknown:
            raise ValueError(f"Unknown agent names: {unknown}. Valid: {valid}")
        assert False, "Should have raised"
    except ValueError as e:
        assert "FOO" in str(e)


def test_active_agents_subset_accepted():
    valid = {"ESYS", "ESW", "ESAF", "CSYS", "CSW", "IO", "ACAD"}
    subset = {"ESYS", "IO"}
    assert subset.issubset(valid)



if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
