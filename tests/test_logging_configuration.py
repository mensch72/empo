import logging
import ast
from pathlib import Path

import pytest

from empo.util.logging import configure_logging


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_configure_logging_uses_env_var(monkeypatch):
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers[:]
    original_level = root_logger.level
    try:
        root_logger.handlers.clear()
        monkeypatch.setenv("EMPO_LOG_LEVEL", "DEBUG")
        configure_logging()
        assert root_logger.level == logging.DEBUG
    finally:
        root_logger.handlers.clear()
        root_logger.handlers.extend(original_handlers)
        root_logger.setLevel(original_level)


def test_configure_logging_rejects_non_level_logging_attributes():
    with pytest.raises(ValueError):
        configure_logging("BASIC_FORMAT")


def _find_print_calls(tree: ast.AST):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "print":
            yield node
        elif (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "print"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "builtins"
        ):
            yield node


def test_src_tree_has_no_print_calls():
    for relative_path in ("src/empo", "src/llm_hierarchical_modeler"):
        for path in (PROJECT_ROOT / relative_path).rglob("*.py"):
            tree = ast.parse(path.read_text(), filename=str(path))
            assert not any(_find_print_calls(tree)), f"Unexpected print() call in {path}"
