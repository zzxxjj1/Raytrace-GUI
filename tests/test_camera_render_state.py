import ast
from pathlib import Path


SOURCE = Path(__file__).parents[1] / "raytrace_v4.py"


def _update_canvas_node():
    module = ast.parse(SOURCE.read_text(encoding="utf-8"))
    return next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "update_canvas"
    )


def test_update_canvas_receives_camera_explicitly():
    update_canvas = _update_canvas_node()

    assert [argument.arg for argument in update_canvas.args.args] == ["camera"]
    assert not any(
        "camera" in node.names
        for node in ast.walk(update_canvas)
        if isinstance(node, ast.Global)
    )
