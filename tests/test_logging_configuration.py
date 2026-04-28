import logging
import tokenize
from io import StringIO
from pathlib import Path

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


def test_src_tree_has_no_print_calls():
    for relative_path in ('src/empo', 'src/llm_hierarchical_modeler'):
        for path in (PROJECT_ROOT / relative_path).rglob('*.py'):
            tokens = tokenize.generate_tokens(StringIO(path.read_text()).readline)
            assert all(
                not (token.type == tokenize.NAME and token.string == "print")
                for token in tokens
            ), f"Unexpected print() in {path}"
