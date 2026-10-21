import sys
from pathlib import Path

# Add the sparse-kernels root to sys.path so that 'formats', 'patterns', 'bench' etc.
# can be imported as top-level packages during test execution.
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "workload: marks tests requiring the extracted BERT intermediate weight tensor"
    )

