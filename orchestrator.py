"""
orchestrator.py
───────────────
Main iterative multi-agent loop.

Logging
───────
All application logs go to both the terminal (INFO) and run.log (DEBUG).
LLM exchange logs go exclusively to llm_exchanges.log (see agents/base.py).
Both files are opened in append mode so successive runs accumulate.

Crash safety
────────────
The graph is saved after every iteration.  If an unhandled exception occurs
the current graph and refs are written to <output>_crash.json / _crash_refs.txt
before re-raising so no work is lost.

Token budget
────────────
Agents receive a compact graph summary (id + name + outputs) instead of the
full JSON, drastically reducing prompt size.  Full task detail is only sent
when an agent refines a specific task.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from pathlib import Path

from core.model import ProcessGraph, ReferenceRegistry, ArtifactRegistry
from rag.pipeline import RAGPipeline
from agents.engineering import ESYSAgent, ESAFAgent, ESWAgent
from agents.refinement import CSYSAgent, CSWAgent
from agents.consistency import IOAgent, ACADAgent, DESCAgent


# ── logging setup (called once) ───────────────────────────────────────────────
def _configure_logging(log_path: str = "run.log") -> logging.Logger:
    """
    Configure the root logger to write to both terminal and *log_path*.

    - Terminal : INFO and above
    - File     : DEBUG and above (append mode)

    Returns the module-level logger for orchestrator.py.
    """
    root = logging.getLogger()
    if root.handlers:
        # Already configured (e.g. module imported twice in tests).
        return logging.getLogger(__name__)

    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # Terminal handler — INFO only (keep the console readable)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    # File handler — DEBUG (full detail, append across runs)
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    return logging.getLogger(__name__)


logger = _configure_logging()


# ── default seed process ──────────────────────────────────────────────────────
_SEED = {
    "process": {
        "inputs": [
            "Aircraft-Level Operational Requirements",
            "Applicable Regulations",
        ],
        "outputs": [
            "Software Accomplishment Summary (SAS)",
            "SSA Report",
        ],
        "tasks": [],
    }
}


# ── save helpers ──────────────────────────────────────────────────────────────
def _save_graph(graph: ProcessGraph, path: str) -> None:
    Path(path).write_text(graph.to_json(), encoding="utf-8")
    logger.debug("Graph saved to '%s' (%d tasks).", path, len(graph.tasks))


def _save_refs(registry: ReferenceRegistry, path: str) -> None:
    registry.save(path)
    logger.debug("Refs saved to '%s' (%d entries).", path, len(registry))


# ── main run function ─────────────────────────────────────────────────────────
def run(
    initial_json: dict | None = None,
    docs_dir: str = "Docs",
    max_iterations: int = 10,
    output_path: str = "process_output.json",
    model: str = "gpt-4o",
    reset_acad: bool = False,
    active_agents:   set[str] | None = None,
    max_refinements: int | None = None,
    graphviz_output: str | None = None,
    graphviz_fmt:    str = "svg",
) -> ProcessGraph:
    """
    Run the multi-agent loop until convergence or max_iterations.

    Parameters
    ----------
    initial_json : dict | None
        Seed process graph. Uses a minimal default if None.
    docs_dir : str
        Directory containing PDF reference documents for RAG.
    max_iterations : int
        Hard stop. Also stops early when no agent changes the graph.
    output_path : str
        Final JSON output path.  Intermediate saves use the same stem.
    model : str
        OpenAI model name.
    """
    refs_path   = output_path.replace(".json", "_refs.txt")
    crash_path  = output_path.replace(".json", "_crash.json")
    crash_refs  = output_path.replace(".json", "_crash_refs.txt")

    # ── initialise ────────────────────────────────────────────────────────────
    seed  = initial_json or _SEED
    graph = ProcessGraph.from_dict(seed)
    logger.info("Initialised graph: %d tasks, %d process inputs.",
                len(graph.tasks), len(graph.inputs))

    rag      = RAGPipeline(docs_dir=docs_dir)
    rag.build()

    registry = ReferenceRegistry.load(refs_path)   # empty if first run
    if reset_acad:
        registry.reset()
        logger.info("Academic reference registry reset (--reset-acad).")
    else:
        logger.info("Registry loaded: %d existing references.", len(registry))

    arts_path = output_path.replace(".json", "_artifacts.txt")
    art_reg   = ArtifactRegistry.load(arts_path)
    logger.info("Artifact registry loaded: %d existing artifacts.", len(art_reg))

    # Migrate seed / loaded graph: replace plain names with ART-xxx IDs
    if graph.migrate_artifacts(art_reg):
        logger.info("Migrated seed graph artifact names to ART-xxx IDs.")
    _save_graph(graph, output_path)
    art_reg.save(arts_path)

    agent_kwargs = dict(graph=graph, rag=rag, registry=registry,
                        art_reg=art_reg, model=model,
                        max_refinements=max_refinements)

    esys = ESYSAgent(**agent_kwargs)
    esaf = ESAFAgent(**agent_kwargs)
    esw  = ESWAgent(**agent_kwargs)
    csys = CSYSAgent(**agent_kwargs)
    csw  = CSWAgent(**agent_kwargs)
    io   = IOAgent(**agent_kwargs)
    desc = DESCAgent(**agent_kwargs)
    acad = ACADAgent(**agent_kwargs)

    all_agents = [esys, esaf, esw, csys, csw, io, desc, acad]

    def _sync():
        for agent in all_agents:
            agent.graph = graph

    # ── agent filter ─────────────────────────────────────────────────────────
    _ALL_AGENTS = {"ESYS", "ESW", "ESAF", "CSYS", "CSW", "IO", "DESC", "ACAD"}
    if active_agents is not None:
        unknown = active_agents - _ALL_AGENTS
        if unknown:
            raise ValueError(
                f"Unknown agent names: {unknown}. Valid: {_ALL_AGENTS}"
            )
        logger.info("Active agents: %s", sorted(active_agents))
    else:
        logger.info("Active agents: all")

    def _enabled(agent) -> bool:
        """Return True if this agent should run this iteration."""
        return active_agents is None or agent.name in active_agents

    # ── iterative loop ────────────────────────────────────────────────────────
    try:
        for iteration in range(1, max_iterations + 1):
            logger.info("=" * 60)
            logger.info("ITERATION %d / %d  |  tasks=%d  inputs=%d",
                        iteration, max_iterations,
                        len(graph.tasks), len(graph.inputs))

            any_change = False

            # Reset per-iteration refinement counters for engineering agents
            if max_refinements is not None:
                for agent in (esys, esw, esaf):
                    agent.reset_refinement_count()

            # Step 1 — Engineering (extend + refine)
            logger.info("--- Step 1: Engineering agents ---")
            for agent in (esys, esw, esaf):
                if _enabled(agent) and agent.run():
                    any_change = True
            _sync()

            # Step 2 — Certification
            logger.info("--- Step 2: Certification agents ---")
            for agent in (csys, csw):
                if _enabled(agent) and agent.run():
                    any_change = True
            _sync()

            # Step 3 — Consistency
            logger.info("--- Step 3: Consistency agent ---")
            if _enabled(io) and io.run():
                any_change = True
            _sync()

            # Step 4 — Academic enrichment
            logger.info("--- Step 4: Academic enrichment ---")
            if acad.run():
                any_change = True
            _sync()
            # Save refs and artifacts immediately after ACAD completes
            _save_refs(registry, refs_path)
            art_reg.save(arts_path)

            # Convergence check
            report = graph.convergence_report(art_reg)
            logger.info("Convergence:\n%s", report)

            # Periodic graph save after every iteration
            _save_graph(graph, output_path)

            if not any_change:
                logger.info("No changes in iteration %d — converged.", iteration)
                break

        else:
            logger.warning("Reached max_iterations=%d.", max_iterations)

    except BaseException as _exc:
        # ── dump on any exit (Ctrl-C, crash, or any other signal) ───────────
        is_interrupt = isinstance(_exc, KeyboardInterrupt)
        if is_interrupt:
            logger.warning("KeyboardInterrupt — dumping all files before exit.")
        else:
            logger.error("Unhandled exception — saving crash snapshot.")
            logger.error(traceback.format_exc())
        try:
            _save_graph(graph, crash_path)
            _save_refs(registry, crash_refs)
            art_reg.save(output_path.replace(".json", "_crash_artifacts.txt"))
            if not is_interrupt and graphviz_output:
                # Also render the graph snapshot if Graphviz was requested
                try:
                    render_graph(graph, art_reg=art_reg,
                                 output_stem=crash_path.replace(".json", ""),
                                 fmt=graphviz_fmt)
                except Exception:
                    pass
            level = logger.warning if is_interrupt else logger.error
            level("Snapshot saved: '%s', '%s'.", crash_path, crash_refs)
        except Exception as save_err:
            logger.error("Could not write snapshot: %s", save_err)
        raise

    # ── final save ────────────────────────────────────────────────────────────
    _save_graph(graph, output_path)
    _save_refs(registry, refs_path)
    art_reg.save(arts_path)
    logger.info(
        "Done. tasks=%d  refs=%d  output='%s'",
        len(graph.tasks), len(registry), output_path,
    )

    # ── optional Graphviz rendering ───────────────────────────────────────
    if graphviz_output is not None:
        stem = graphviz_output
        if stem.endswith("."):
            stem = stem.rstrip(".")
        # Strip extension if user included one so we control it cleanly
        for ext in (".gv", ".dot", ".svg", ".pdf", ".png"):
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break
        gv_path = render_graph(
            graph,
            art_reg=art_reg,
            output_stem=stem,
            fmt=graphviz_fmt,
        )
        logger.info("Graphviz source: '%s'.", gv_path)

    return graph


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Aeronautical multi-agent process generator"
    )
    parser.add_argument("--input",    "-i", default=None)
    parser.add_argument("--docs",     "-d", default="Docs")
    parser.add_argument("--output",   "-o", default="process_output.json")
    parser.add_argument("--max-iter", "-n", type=int, default=10)
    parser.add_argument("--model",    "-m", default="gpt-4o")
    parser.add_argument(
        "--agents", "-a", default=None,
        help="Comma-separated agent names to run (default: all). "
             "Valid: ESYS,ESW,ESAF,CSYS,CSW,IO,ACAD"
    )
    parser.add_argument(
        "--max-refine", type=int, default=None,
        help="Max number of task decompositions per agent (default: unlimited)"
    )
    parser.add_argument(
        "--graphviz", default=None, metavar="STEM",
        help="Generate Graphviz output at STEM.gv / STEM.<fmt>"
    )
    parser.add_argument(
        "--graphviz-fmt", default="svg",
        help="Graphviz output format: svg, pdf, png, etc. (default: svg)"
    )
    parser.add_argument("--reset-acad", action="store_true",
                        help="Clear academic references at startup")
    parser.add_argument("--log",            default="run.log",
                        help="Application log file path")
    args = parser.parse_args()

    # Re-configure logging with the user-specified log path
    # (the default call at import time used "run.log")
    logging.getLogger().handlers.clear()
    _configure_logging(args.log)

    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY not set.")

    initial = None
    if args.input:
        with open(args.input, encoding="utf-8") as f:
            initial = json.load(f)

    run(
        initial_json=initial,
        docs_dir=args.docs,
        max_iterations=args.max_iter,
        output_path=args.output,
        model=args.model,
        reset_acad=args.reset_acad,
        active_agents=(
            {a.strip().upper() for a in args.agents.split(",")}
            if args.agents else None
        ),
        max_refinements=args.max_refine,
        graphviz_output=args.graphviz,
        graphviz_fmt=args.graphviz_fmt,
    )
