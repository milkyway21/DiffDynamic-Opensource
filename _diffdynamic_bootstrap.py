"""Make the source package available to historical checkout entry points."""

import sys
from pathlib import Path


def bootstrap():
    root = Path(__file__).resolve().parent
    for path in (root, root / "src"):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    return root


bootstrap()
