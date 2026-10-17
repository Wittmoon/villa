"""Copy (infer_rowcol_triplet_wraps) must overlap its inference blocks by default.

With --bbox-overlap 0.0 the blocks abut, and the copied surface jumps by half a
sheet at every block edge (issue #2040). The module needs torch to import, so
the default is read from the source instead of by calling parse_args.
"""

import ast
from pathlib import Path

SOURCE = (
    Path(__file__).resolve().parents[2]
    / "src/vesuvius/neural_tracing/inference/infer_rowcol_triplet_wraps.py"
)


def _bbox_overlap_default():
    for node in ast.walk(ast.parse(SOURCE.read_text())):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "attr", "") == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "--bbox-overlap"
        ):
            return next(k.value.value for k in node.keywords if k.arg == "default")
    raise AssertionError("--bbox-overlap argument not found")


def test_bbox_overlap_default_leaves_no_block_edge_seams():
    assert _bbox_overlap_default() == 0.5
