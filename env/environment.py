from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

from .graders import grade_task
from .models import Action, ActionType, Category, Observation, Priority, State, Team
from .reward import compute_reward
from .tasks import TASKS, TaskDefinition, get_task


class CustomerSupportEnv:
    """Deterministic customer support workflow environment."""

    def __init__(self) -> None:
        self._state: Optional[State] = None
        self._task: Optional[TaskDefinition] = None
        self._latest_score: float = 0.0

    def reset(self, task_id: str = "easy_password_reset") -> Observation:
        task = get_task(task_id)
        self._task = task
        self._state = State(
            task_id=task.task_id,
            objective=task.objective,
            ticket=deepcopy(task.ticket),
            max_steps=task.max_steps,
        )
        self._latest_score = 0.0
        return self._observation()

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        if self._state is None or self._task is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        if self._state.resolved:
            obs = self._observation()
            return obs, 0.0, True, {"score": self._latest_score, "message": "episode already done"}

        self._state.step_count += 1
        action_sig = self._action_signature(action)
        self._state.action_history.append(action_sig)

        repeated_action = self._state.action_history.count(action_sig) > 2
        invalid_action, error_msg = self._apply_action(action)
        self._state.last_action_error = error_msg

        score_before = self._latest_score
        current_score, components = grade_task(self._state, self._task)
        self._latest_score = current_score

        done = self._state.resolved or self._state.step_count >= self._state.max_steps
        reward, breakdown = compute_reward(
            score_before,
            current_score,
            invalid_action=invalid_action,
            repeated_action=repeated_action,
            done=done,
        )

        info = {
            "task_id": self._task.task_id,
            "score": current_score,
            "grader_components": components,
            "reward_breakdown": breakdown,
        }

        return self._observation(), reward, done, info

    def state(self) -> State:
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self._state.model_copy(deep=True)

    def available_tasks(self) -> list[str]:
        return list(TASKS.keys())

    def _observation(self) -> Observation:
        assert self._state is not None
        return Observation(
            task_id=self._state.task_id,
            objective=self._state.objective,
            ticket_snapshot={
                "ticket_id": self._state.ticket.ticket_id,
                "customer_tier": self._state.ticket.customer_tier,
                "subject": self._state.ticket.subject,
                "description": self._state.ticket.description,
                "sla_minutes": self._state.ticket.sla_minutes,
                "order_amount": self._state.ticket.order_amount,
            },
            workspace={
                "priority": self._state.priority.value if self._state.priority else None,
                "category": self._state.category.value if self._state.category else None,
                "internal_notes_count": len(self._state.internal_notes),
                "requested_info_count": len(self._state.requested_info),
                "reply_ready": bool(self._state.composed_reply),
                "refund_amount": self._state.refund_amount,
                "escalated_team": self._state.escalated_team.value if self._state.escalated_team else None,
                "resolved": self._state.resolved,
            },
            step_count=self._state.step_count,
            max_steps=self._state.max_steps,
            last_action_error=self._state.last_action_error,
            allowed_actions=[a.value for a in ActionType],
        )

    def _action_signature(self, action: Action) -> str:
        return f"{action.action_type.value}|{action.value or ''}|{action.amount if action.amount is not None else ''}"

    def _apply_action(self, action: Action) -> Tuple[bool, Optional[str]]:
        assert self._state is not None
        assert self._task is not None

        error: Optional[str] = None

        if action.action_type == ActionType.VIEW_TICKET:
            return False, None

        if action.action_type == ActionType.SET_PRIORITY:
            try:
                self._state.priority = Priority(action.value or "")
            except ValueError:
                error = f"invalid priority: {action.value}"

        elif action.action_type == ActionType.SET_CATEGORY:
            try:
                self._state.category = Category(action.value or "")
            except ValueError:
                error = f"invalid category: {action.value}"

        elif action.action_type == ActionType.ADD_INTERNAL_NOTE:
            note = (action.value or "").strip()
            if not note:
                error = "internal note cannot be empty"
            else:
                self._state.internal_notes.append(note)

        elif action.action_type == ActionType.REQUEST_MISSING_INFO:
            msg = (action.value or "").strip()
            if not msg:
                error = "requested info prompt cannot be empty"
            else:
                self._state.requested_info.append(msg)

        elif action.action_type == ActionType.COMPOSE_REPLY:
            reply = (action.value or "").strip()
            if len(reply) < 20:
                error = "reply too short; must be at least 20 chars"
            else:
                self._state.composed_reply = reply

        elif action.action_type == ActionType.OFFER_REFUND:
            amount = float(action.amount or 0.0)
            if amount <= 0:
                error = "refund amount must be > 0"
            elif self._state.ticket.order_amount <= 0:
                error = "ticket has no refundable amount"
            elif amount > self._state.ticket.order_amount:
                error = "refund exceeds order amount"
            else:
                self._state.refund_amount = round(amount, 2)

        elif action.action_type == ActionType.ESCALATE_TO_TEAM:
            try:
                self._state.escalated_team = Team(action.value or "")
            except ValueError:
                error = f"invalid team: {action.value}"

        elif action.action_type == ActionType.RESOLVE_TICKET:
            missing = []
            if not self._state.priority:
                missing.append("priority")
            if not self._state.category:
                missing.append("category")
            if not self._state.composed_reply:
                missing.append("composed_reply")
            if missing:
                error = f"cannot resolve; missing {','.join(missing)}"
            else:
                self._state.resolved = True

        else:
            error = f"unsupported action_type: {action.action_type}"

        if error:
            return True, error

        return False, None
