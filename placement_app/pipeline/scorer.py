"""
Score and rank candidate placement positions.
"""

import time
from typing import List, Dict, Optional
from PIL import Image

from .composite import make_composite


class PlacementScorer:
    """Evaluate a list of candidate placements and return ranked results.

    Args:
        model:  A ``SimOPAScorer`` instance (or any object with a
                ``score(composite, mask) -> float`` method).
    """

    def __init__(self, model):
        self.model = model

    def score_candidates(self, bg: Image.Image, fg: Image.Image,
                         fg_mask: Image.Image,
                         candidates: List[Dict]) -> List[Dict]:
        """
        Args:
            bg:          Background PIL image (RGB).
            fg:          Foreground PIL image (RGB).
            fg_mask:     Foreground mask (L).
            candidates:  List of dicts from ``generate_candidates``.

        Returns:
            The same list with added keys:
                ``score``     — float in [0, 1], higher = better.
                ``composite`` — PIL Image of the composited result.
                ``mask``      — PIL Image (L) of the foreground mask.
            Sorted by score descending.
        """
        results = []
        for cand in candidates:
            composite, mask = make_composite(bg, fg, fg_mask, cand['bbox'])
            score = self.model.score(composite, mask)
            results.append({
                **cand,
                'score':     score,
                'composite': composite,
                'mask':      mask,
            })

        results.sort(key=lambda r: r['score'], reverse=True)
        return results

    def score_single(self, bg: Image.Image, fg: Image.Image,
                     fg_mask: Image.Image,
                     bbox: list) -> Dict:
        """Score a single bbox and return the result dict."""
        composite, mask = make_composite(bg, fg, fg_mask, bbox)
        score = self.model.score(composite, mask)
        return {
            'bbox':      bbox,
            'score':     score,
            'composite': composite,
            'mask':      mask,
            'scale':     1.0,
        }

    def compare_models(self, bg: Image.Image, fg: Image.Image,
                       fg_mask: Image.Image,
                       candidates: List[Dict],
                       other_model) -> List[Dict]:
        """
        Score candidates with BOTH this model and another model, returning
        results with both scores for comparison.

        Args:
            other_model: Another scorer with the same ``score()`` interface.

        Returns:
            List of dicts with ``score`` (this model), ``score_other``,
            sorted by ``score``.
        """
        results = self.score_candidates(bg, fg, fg_mask, candidates)
        for r in results:
            r['score_other'] = other_model.score(r['composite'], r['mask'])
        return results
