"""외부 앱(Streamlit 등)에서 파이프라인을 import할 때 사용. ``main`` 이름 충돌을 피하기 위한 진입점."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from main import run_pipeline  # noqa: E402

__all__ = ["run_pipeline"]
