from __future__ import annotations

from typing import Dict, Tuple

from .models import RewardBreakdown


def compute_reward(
    previous_score: float,
    current_score: float,
    *,
    invalid_action: bool,
    repeated_action: bool,
    done: bool,
) -> Tuple[float, Dict[str, float]]:
    progress_delta = max(current_score - previous_score, 0.0)
    step_penalty = -0.01
    invalid_action_penalty = -0.08 if invalid_action else 0.0
    loop_penalty = -0.03 if repeated_action else 0.0
    completion_bonus = 0.25 * current_score if done else 0.0

    total = progress_delta + step_penalty + invalid_action_penalty + loop_penalty + completion_bonus
    total = max(min(total, 1.0), -1.0)

    breakdown = RewardBreakdown(
        total=round(total, 4),
        progress_delta=round(progress_delta, 4),
        step_penalty=round(step_penalty, 4),
        invalid_action_penalty=round(invalid_action_penalty, 4),
        loop_penalty=round(loop_penalty, 4),
        completion_bonus=round(completion_bonus, 4),
    )
    return breakdown.total, breakdown.model_dump()
