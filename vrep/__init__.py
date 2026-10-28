"""vrep: what makes a visual representation predictable under action?"""

import os

# Let unsupported ops fall back to CPU on Apple silicon (must be set before torch initialises MPS).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

__version__ = "0.1.0"
