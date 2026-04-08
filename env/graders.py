from __future__ import annotations

from typing import Dict, Tuple

from .models import State
from .tasks import TaskDefinition


def _contains_keywords(text: str, keywords: list[str]) -> bool:
    lower = text.lower()
    return all(k.lower() in lower for k in keywords)


def grade_task(state: State, task: TaskDefinition) -> Tuple[float, Dict[str, float]]:
    if task.task_id == "easy_password_reset":
        return _grade_easy(state, task)
    if task.task_id == "medium_billing_duplicate":
        return _grade_medium(state, task)
    if task.task_id == "hard_enterprise_outage":
        return _grade_hard(state, task)
    return 0.0, {"unknown_task": 0.0}


def _grade_easy(state: State, task: TaskDefinition) -> Tuple[float, Dict[str, float]]:
    components = {
        "category": 1.0 if state.category and state.category.value == "account" else 0.0,
        "priority": 1.0 if state.priority and state.priority.value == "medium" else 0.0,
        "reply_quality": 0.0,
        "resolved": 1.0 if state.resolved else 0.0,
        "no_refund": 1.0 if state.refund_amount == 0 else 0.0,
    }

    if state.composed_reply:
        components["reply_quality"] = 1.0 if _contains_keywords(state.composed_reply, task.ticket.required_keywords) else 0.0

    score = (
        0.20 * components["category"]
        + 0.20 * components["priority"]
        + 0.35 * components["reply_quality"]
        + 0.20 * components["resolved"]
        + 0.05 * components["no_refund"]
    )
    return round(min(max(score, 0.0), 1.0), 4), components


def _grade_medium(state: State, task: TaskDefinition) -> Tuple[float, Dict[str, float]]:
    required_refund = float(task.constraints.get("required_refund", "0"))
    components = {
        "priority": 1.0 if state.priority and state.priority.value == "high" else 0.0,
        "category": 1.0 if state.category and state.category.value == "billing" else 0.0,
        "internal_note": 1.0 if len(state.internal_notes) > 0 else 0.0,
        "refund_exact": 1.0 if abs(state.refund_amount - required_refund) < 1e-9 else 0.0,
        "reply_quality": 0.0,
        "resolved": 1.0 if state.resolved else 0.0,
    }

    if state.composed_reply:
        components["reply_quality"] = 1.0 if _contains_keywords(state.composed_reply, task.ticket.required_keywords) else 0.0

    score = (
        0.15 * components["priority"]
        + 0.20 * components["category"]
        + 0.15 * components["internal_note"]
        + 0.20 * components["refund_exact"]
        + 0.20 * components["reply_quality"]
        + 0.10 * components["resolved"]
    )
    return round(min(max(score, 0.0), 1.0), 4), components


def _grade_hard(state: State, task: TaskDefinition) -> Tuple[float, Dict[str, float]]:
    components = {
        "priority": 1.0 if state.priority and state.priority.value == "urgent" else 0.0,
        "category": 1.0 if state.category and state.category.value == "technical" else 0.0,
        "escalation": 1.0 if state.escalated_team and state.escalated_team.value == "tech_l2" else 0.0,
        "internal_note": 1.0 if len(state.internal_notes) > 0 else 0.0,
        "requested_info": 1.0 if len(state.requested_info) > 0 else 0.0,
        "reply_quality": 0.0,
        "resolved": 1.0 if state.resolved else 0.0,
    }

    if state.composed_reply:
        components["reply_quality"] = 1.0 if _contains_keywords(state.composed_reply, task.ticket.required_keywords) else 0.0

    score = (
        0.15 * components["priority"]
        + 0.15 * components["category"]
        + 0.20 * components["escalation"]
        + 0.10 * components["internal_note"]
        + 0.10 * components["requested_info"]
        + 0.20 * components["reply_quality"]
        + 0.10 * components["resolved"]
    )
    return round(min(max(score, 0.0), 1.0), 4), components
