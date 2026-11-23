"""Repository-level pytest guard.

`tests/unit/test_harvest_logic.py` installs a lightweight `dolfinx` mock into
`sys.modules` during collection.  `tests/unit/test_physics_model.py` relies on
`pytest.importorskip("dolfinx")`, which then sees that mock instead of the
real (absent) FEniCSx installation and tries to import the scientific FEM
module.  When FEniCSx is genuinely unavailable, honor the test's own intent
and skip the file at collection time.  No existing test file is modified.
"""

from __future__ import annotations

import importlib.util


def pytest_ignore_collect(collection_path, config):  # noqa: ANN001, ARG001
    if collection_path.name != "test_physics_model.py":
        return None
    try:
        if importlib.util.find_spec("dolfinx") is None:
            return True
    except (ImportError, ValueError):
        return True
    return None
