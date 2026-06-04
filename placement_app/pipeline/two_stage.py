"""
Two-stage placement pipeline:  coarse grid → fine local search.

This approximates the "FOPA → SimOPA" two-stage idea without requiring
the full libcom / FOPA dependencies:

  Stage 1 — Coarse:  large-step grid (e.g. 7×7, 1 scale)   → 25–49 candidates
  Stage 2 — Fine:    local random search around Top-K       → N×15 more candidates

The two-stage approach gives better placement recommendations than a
single dense grid because the model can refine positions near promising
regions.
"""

import random
from typing import List, Dict
from PIL import Image

from .candidates import generate_candidates
from .scorer import PlacementScorer


class TwoStageScorer:
    """Coarse-to-fine placement scorer.

    Args:
        model:             SimOPAScorer instance.
        coarse_grid:       Grid density for stage 1.
        top_k:             How many coarse candidates to refine.
        fine_radius:       Search radius (pixels) around each coarse candidate.
        fine_samples:      Number of random samples per coarse candidate.
    """

    def __init__(self, model,
                 coarse_grid: int = 7,
                 top_k: int = 3,
                 fine_radius: int = 30,
                 fine_samples: int = 15):
        self.pipe = PlacementScorer(model)
        self.coarse_grid = coarse_grid
        self.top_k = top_k
        self.fine_radius = fine_radius
        self.fine_samples = fine_samples

    def score(self, bg: Image.Image, fg: Image.Image,
              fg_mask: Image.Image) -> List[Dict]:
        """
        Run the two-stage pipeline.

        Returns:
            List of all scored candidates (coarse + fine), sorted by score.
            Each dict contains ``stage`` ('coarse' | 'fine').
        """
        bg_w, bg_h = bg.size
        fg_w, fg_h = fg.size

        # ---- Stage 1: coarse grid ----
        coarse = generate_candidates(bg_w, bg_h, fg_w, fg_h,
                                      grid_size=self.coarse_grid, n_scales=1)
        if not coarse:
            return []

        coarse_results = self.pipe.score_candidates(bg, fg, fg_mask, coarse)
        for r in coarse_results:
            r['stage'] = 'coarse'

        # ---- Stage 2: local refinement around top-K ----
        fine_candidates = []
        for rank, top in enumerate(coarse_results[:self.top_k]):
            x1, y1, x2, y2 = top['bbox']
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            for _ in range(self.fine_samples):
                dx = random.randint(-self.fine_radius, self.fine_radius)
                dy = random.randint(-self.fine_radius, self.fine_radius)
                nx1 = max(0, min(cx + dx - fg_w // 2, bg_w - fg_w))
                ny1 = max(0, min(cy + dy - fg_h // 2, bg_h - fg_h))
                fine_candidates.append({
                    'bbox':  [nx1, ny1, nx1 + fg_w, ny1 + fg_h],
                    'scale': 1.0,
                    'refine_from': rank + 1,
                })

        if fine_candidates:
            fine_results = self.pipe.score_candidates(
                bg, fg, fg_mask, fine_candidates)
            for r in fine_results:
                r['stage'] = 'fine'
        else:
            fine_results = []

        # Merge and sort
        all_results = coarse_results + fine_results
        all_results.sort(key=lambda r: r['score'], reverse=True)
        return all_results

    def generate_comparison(self, bg, fg, fg_mask) -> dict:
        """
        Run BOTH single-stage dense grid and two-stage, return comparison
        data for analysis.

        Returns:
            dict with 'single' and 'two_stage' result lists, plus timing.
        """
        import time
        bg_w, bg_h = bg.size
        fg_w, fg_h = fg.size

        # Single-stage baseline: dense grid (equivalent total budget)
        n_fine = self.top_k * self.fine_samples
        n_coarse = self.coarse_grid ** 2
        total_budget = n_coarse + n_fine
        single_grid = max(2, int(total_budget ** 0.5))
        single_candidates = generate_candidates(bg_w, bg_h, fg_w, fg_h,
                                                 grid_size=single_grid,
                                                 n_scales=1)[:total_budget]

        t0 = time.time()
        single_results = self.pipe.score_candidates(bg, fg, fg_mask,
                                                     single_candidates)
        single_time = time.time() - t0

        t0 = time.time()
        two_stage_results = self.score(bg, fg, fg_mask)
        two_stage_time = time.time() - t0

        return {
            'single':     single_results,
            'two_stage':  two_stage_results,
            'single_time': single_time,
            'two_stage_time': two_stage_time,
            'budget': total_budget,
        }
