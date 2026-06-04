"""
Placement scoring pipeline.

- ``candidates`` : generate candidate bounding boxes on a background.
- ``composite``  : paste a foreground onto a background at a given bbox.
- ``scorer``     : evaluate multiple candidates with a SimOPAScorer and rank them.
"""

from .candidates import generate_candidates
from .composite import make_composite
from .scorer import PlacementScorer
