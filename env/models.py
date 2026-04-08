from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class ActionType(str, Enum):
    VIEW_TICKET = "view_ticket"
    SET_PRIORITY = "set_priority"
    SET_CATEGORY = "set_category"
    ADD_INTERNAL_NOTE = "add_internal_note"
    REQUEST_MISSING_INFO = "request_missing_info"
    COMPOSE_REPLY = "compose_reply"
    OFFER_REFUND = "offer_refund"
    ESCALATE_TO_TEAM = "escalate_to_team"
    RESOLVE_TICKET = "resolve_ticket"


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class Category(str, Enum):
    ACCOUNT = "account"
    BILLING = "billing"
    TECHNICAL = "technical"
    SHIPPING = "shipping"


class Team(str, Enum):
    BILLING_OPS = "billing_ops"
    TECH_L2 = "tech_l2"
    RISK = "risk"


class Ticket(BaseModel):
    ticket_id: str
    customer_tier: str
    subject: str
    description: str
    order_amount: float = 0.0
    sla_minutes: int = 240
    required_keywords: List[str] = Field(default_factory=list)


class Action(BaseModel):
    action_type: ActionType
    value: Optional[str] = None
    amount: Optional[float] = None

    @model_validator(mode="after")
    def validate_payload(self) -> "Action":
        if self.action_type in {
            ActionType.SET_PRIORITY,
            ActionType.SET_CATEGORY,
            ActionType.ADD_INTERNAL_NOTE,
            ActionType.REQUEST_MISSING_INFO,
            ActionType.COMPOSE_REPLY,
            ActionType.ESCALATE_TO_TEAM,
        } and not self.value:
            raise ValueError(f"action '{self.action_type}' requires non-empty value")

        if self.action_type == ActionType.OFFER_REFUND and self.amount is None:
            raise ValueError("action 'offer_refund' requires amount")

        return self


class Observation(BaseModel):
    task_id: str
    objective: str
    ticket_snapshot: Dict[str, Any]
    workspace: Dict[str, Any]
    step_count: int
    max_steps: int
    last_action_error: Optional[str] = None
    allowed_actions: List[str]


class State(BaseModel):
    task_id: str
    objective: str
    ticket: Ticket
    priority: Optional[Priority] = None
    category: Optional[Category] = None
    internal_notes: List[str] = Field(default_factory=list)
    requested_info: List[str] = Field(default_factory=list)
    composed_reply: Optional[str] = None
    refund_amount: float = 0.0
    escalated_team: Optional[Team] = None
    resolved: bool = False
    step_count: int = 0
    max_steps: int = 10
    action_history: List[str] = Field(default_factory=list)
    last_action_error: Optional[str] = None


class RewardBreakdown(BaseModel):
    total: float
    progress_delta: float
    step_penalty: float
    invalid_action_penalty: float
    loop_penalty: float
    completion_bonus: float
