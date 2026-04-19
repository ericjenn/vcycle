"""
core/viz.py
───────────
Graphviz renderer for the aeronautical process graph.

Visual conventions
──────────────────
Domain colour coding (fill colour of task nodes):
  System   (ARP4754A) — steel blue      #4472C4 / light: #BDD7EE
  Safety   (ARP4761)  — coral / red     #FF0000 / light: #FFCCCC  (actually salmon)
  Software (DO-178C)  — green           #70AD47 / light: #E2EFDA
  Unknown / mixed     — light grey      #F2F2F2

Hierarchy:
  Parent tasks are rendered as Graphviz *clusters* (subgraphs).
  Each child task appears as a node inside its parent's cluster.
  A dashed edge from the cluster's invisible anchor to each child makes the
  parent→child containment visible even in non-cluster renderers.

Artifact flow:
  A solid directed edge from the task that produces an artifact to the task
  that consumes it, labelled with the artifact ID (resolved to full name when
  an ArtifactRegistry is available).

Output:
  A .gv (DOT source) file and, if graphviz is installed, a .svg / .pdf / .png
  rendered output file.

Usage
─────
    from core.viz import render_graph
    render_graph(graph, art_reg, output_stem="process_output",
                 fmt="svg")          # → process_output.gv + process_output.svg
"""

from __future__ import annotations

import logging
import subprocess
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.model import ProcessGraph, ArtifactRegistry

logger = logging.getLogger(__name__)


# ── domain detection ──────────────────────────────────────────────────────────

# Keywords used to classify tasks into domains.
_SYSTEM_KW = {
    "aircraft", "system", "function", "allocation", "architecture",
    "interface", "arp4754", "arp 4754",
}
_SAFETY_KW = {
    "safety", "hazard", "fha", "pssa", "ssa", "fta", "fmea", "cma",
    "failure", "fault", "arp4761", "arp 4761", "dal",
}
_SOFTWARE_KW = {
    "software", "do-178", "do 178", "hlr", "llr", "source code",
    "verification", "testing", "coverage", "psac", "sdp", "svp",
    "scmp", "sqap", "configuration", "quality", "executable", "coding",
}


def _domain(task) -> str:
    text = (task.name + " " + task.description + " " +
            " ".join(task.standards)).lower()
    # Safety takes precedence over system/software (safety is cross-cutting)
    if any(k in text for k in _SAFETY_KW):
        return "safety"
    if any(k in text for k in _SOFTWARE_KW):
        return "software"
    if any(k in text for k in _SYSTEM_KW):
        return "system"
    return "unknown"


# ── colour palette ────────────────────────────────────────────────────────────

_DOMAIN_STYLE: dict[str, dict[str, str]] = {
    "system": {
        "fillcolor": "#BDD7EE",
        "color":     "#4472C4",
        "fontcolor": "#1F3864",
        "cluster_bg":"#EEF4FB",
    },
    "safety": {
        "fillcolor": "#FFCCCC",
        "color":     "#C00000",
        "fontcolor": "#600000",
        "cluster_bg":"#FFF0F0",
    },
    "software": {
        "fillcolor": "#E2EFDA",
        "color":     "#375623",
        "fontcolor": "#375623",
        "cluster_bg":"#F4FAF0",
    },
    "unknown": {
        "fillcolor": "#F2F2F2",
        "color":     "#767676",
        "fontcolor": "#404040",
        "cluster_bg":"#FAFAFA",
    },
}

# ── DOT helpers ───────────────────────────────────────────────────────────────

def _esc(s: str) -> str:
    """Escape a string for use as a Graphviz label."""
    return s.replace('"', '\\"').replace('\n', '\\n')


def _node_id(task_id: int) -> str:
    return f"task_{task_id}"


def _cluster_id(task_id: int) -> str:
    return f"cluster_{task_id}"


def _wrap(text: str, width: int = 30) -> str:
    """Wrap a label to *width* characters, using \\n as line separator."""
    return "\\n".join(textwrap.wrap(text, width=width))


# ── main renderer ─────────────────────────────────────────────────────────────

def render_graph(
    graph:       "ProcessGraph",
    art_reg:     "ArtifactRegistry | None" = None,
    output_stem: str  = "process_output",
    fmt:         str  = "svg",
    view:        bool = False,
) -> str:
    """
    Generate a Graphviz DOT file and (optionally) render it.

    Parameters
    ----------
    graph : ProcessGraph
        The process graph to render.
    art_reg : ArtifactRegistry | None
        If provided, ART-xxx IDs in edge labels are resolved to full names.
    output_stem : str
        Path stem for output files.  Two files are written:
          ``<stem>.gv``        — DOT source
          ``<stem>.<fmt>``     — rendered image (if graphviz is installed)
    fmt : str
        Output format passed to the ``dot`` command: svg, pdf, png, etc.
    view : bool
        If True and graphviz is installed, open the rendered file in the
        default viewer (via ``xdg-open`` / ``open``).

    Returns
    -------
    str
        Path to the written .gv file.
    """
    dot = _build_dot(graph, art_reg)

    gv_path = output_stem + ".gv"
    Path(gv_path).write_text(dot, encoding="utf-8")
    logger.info("DOT source written to '%s'.", gv_path)

    # Attempt rendering — silently skip if graphviz not installed
    out_path = output_stem + "." + fmt
    try:
        result = subprocess.run(
            ["dot", f"-T{fmt}", gv_path, "-o", out_path],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            logger.info("Graphviz rendered '%s'.", out_path)
            if view:
                import platform, os
                opener = "open" if platform.system() == "Darwin" else "xdg-open"
                subprocess.Popen([opener, out_path])
        else:
            logger.warning(
                "Graphviz rendering failed (exit %d): %s",
                result.returncode, result.stderr[:300],
            )
    except FileNotFoundError:
        logger.warning(
            "graphviz 'dot' not found — only the .gv source was written. "
            "Install with: apt install graphviz  or  brew install graphviz"
        )
    except subprocess.TimeoutExpired:
        logger.warning("Graphviz rendering timed out.")

    return gv_path


# ── DOT builder ───────────────────────────────────────────────────────────────

def _build_dot(
    graph:   "ProcessGraph",
    art_reg: "ArtifactRegistry | None",
) -> str:
    """Return the complete DOT source string."""

    lines: list[str] = []
    a = lines.append   # convenience alias

    a('digraph process {')
    a('  graph [rankdir=LR, splines=ortho, fontname="Helvetica", fontsize=10,')
    a('         label="Aeronautical Development Process", labelloc=t, labeljust=l,')
    a('         bgcolor="#FFFFFF", pad=0.5];')
    a('  node  [shape=box, style="filled,rounded", fontname="Helvetica",')
    a('         fontsize=9, margin="0.12,0.06"];')
    a('  edge  [fontname="Helvetica", fontsize=8, color="#555555"];')
    a('')

    # ── build helper maps ─────────────────────────────────────────────────────
    children_of: dict[int, list] = {t.id: [] for t in graph.tasks}
    for t in graph.tasks:
        if t.parent_id is not None and t.parent_id in children_of:
            children_of[t.parent_id].append(t)

    # Producers: artifact_id → task_id
    producers: dict[str, int] = {}
    for t in graph.tasks:
        for art in t.outputs:
            producers[art] = t.id

    def _art_label(art_id: str) -> str:
        if art_reg:
            name = art_reg.resolve(art_id)
            if name:
                return _wrap(name, 25)
        return art_id

    # ── legend ────────────────────────────────────────────────────────────────
    a('  subgraph cluster_legend {')
    a('    label="Legend"; style=dotted; color="#AAAAAA";')
    a('    fontname="Helvetica"; fontsize=9;')
    for dom, style in _DOMAIN_STYLE.items():
        nid = f"legend_{dom}"
        lbl = dom.capitalize()
        a(f'    {nid} [label="{lbl}", fillcolor="{style["fillcolor"]}", '
          f'color="{style["color"]}", fontcolor="{style["fontcolor"]}", '
          f'width=1.2, height=0.3];')
    a('  }')
    a('')

    # ── process-level boundary nodes ─────────────────────────────────────────
    a('  // Process inputs')
    for art in sorted(set(graph.inputs)):
        nid = f'proc_in_{art.replace("-","_")}'
        a(f'  {nid} [label="{_esc(_art_label(art))}", shape=parallelogram, '
          f'fillcolor="#FFF2CC", color="#D6B656", fontcolor="#7D4E00"];')
    a('')
    a('  // Process outputs')
    for art in sorted(set(graph.outputs)):
        nid = f'proc_out_{art.replace("-","_")}'
        a(f'  {nid} [label="{_esc(_art_label(art))}", shape=parallelogram, '
          f'fillcolor="#D5E8D4", color="#82B366", fontcolor="#1A4A1A"];')
    a('')

    # ── render task tree recursively ──────────────────────────────────────────
    root_tasks = [t for t in graph.tasks if t.parent_id is None]

    def _render_task(task, indent: int) -> None:
        pad  = "  " * indent
        dom  = _domain(task)
        styl = _DOMAIN_STYLE[dom]
        kids = children_of.get(task.id, [])

        if kids:
            # Parent task → render as a cluster
            a(f'{pad}subgraph {_cluster_id(task.id)} {{')
            a(f'{pad}  label="{_esc(_wrap(task.name, 35))}";')
            a(f'{pad}  style=filled; fillcolor="{styl["cluster_bg"]}";')
            a(f'{pad}  color="{styl["color"]}"; fontcolor="{styl["fontcolor"]}";')
            a(f'{pad}  fontname="Helvetica"; fontsize=9;')
            # Invisible anchor node so edges can target the cluster
            anchor = f'anchor_{task.id}'
            a(f'{pad}  {anchor} [style=invis, shape=point, width=0, height=0];')
            for child in kids:
                _render_task(child, indent + 1)
            a(f'{pad}}}')
        else:
            # Leaf task → regular node
            desc_short = _wrap(task.name, 35)
            tooltip    = _esc(task.description[:120]) if task.description else ""
            a(f'{pad}{_node_id(task.id)} ['
              f'label="{_esc(desc_short)}", '
              f'fillcolor="{styl["fillcolor"]}", '
              f'color="{styl["color"]}", '
              f'fontcolor="{styl["fontcolor"]}", '
              f'tooltip="{tooltip}", '
              f'id="task_{task.id}"];')

    for t in root_tasks:
        _render_task(t, 1)

    a('')

    # ── parent → child containment edges (dashed) ─────────────────────────────
    a('  // Parent→child containment links')
    for t in graph.tasks:
        kids = children_of.get(t.id, [])
        if kids:
            for child in kids:
                # Edge from parent anchor to child node (or child anchor)
                src = f'anchor_{t.id}'
                dst = _node_id(child.id) if not children_of.get(child.id) else f'anchor_{child.id}'
                a(f'  {src} -> {dst} '
                  f'[style=dashed, color="{_DOMAIN_STYLE[_domain(t)]["color"]}", '
                  f'arrowhead=open, weight=0];')
    a('')

    # ── artifact flow edges (solid) ───────────────────────────────────────────
    a('  // Artifact flow edges (producer → consumer)')
    for consumer in graph.tasks:
        for art in consumer.inputs:
            producer_id = producers.get(art)
            if producer_id is None:
                # Flows from process input
                src_nid = f'proc_in_{art.replace("-","_")}'
                dst_nid = _node_id(consumer.id) if not children_of.get(consumer.id) \
                          else f'anchor_{consumer.id}'
                a(f'  {src_nid} -> {dst_nid} '
                  f'[label="{_esc(_art_label(art))}", '
                  f'color="#888888", style=solid, arrowsize=0.7];')
            else:
                src_nid = _node_id(producer_id) if not children_of.get(producer_id) \
                          else f'anchor_{producer_id}'
                dst_nid = _node_id(consumer.id) if not children_of.get(consumer.id) \
                          else f'anchor_{consumer.id}'
                a(f'  {src_nid} -> {dst_nid} '
                  f'[label="{_esc(_art_label(art))}", '
                  f'color="#555555", style=solid, arrowsize=0.7];')

    # ── edges from leaf outputs to process outputs ────────────────────────────
    a('  // Task outputs → process boundary outputs')
    proc_outputs = set(graph.outputs)
    for t in graph.tasks:
        for art in t.outputs:
            if art in proc_outputs:
                src_nid = _node_id(t.id) if not children_of.get(t.id) \
                          else f'anchor_{t.id}'
                dst_nid = f'proc_out_{art.replace("-","_")}'
                a(f'  {src_nid} -> {dst_nid} '
                  f'[label="{_esc(_art_label(art))}", '
                  f'color="#82B366", style=solid, arrowsize=0.7];')

    a('}')
    return "\n".join(lines)
