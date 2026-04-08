from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field

from .models import Ticket


class TaskDefinition(BaseModel):
    task_id: str
    level: str
    title: str
    objective: str
    max_steps: int
    required_actions: List[str] = Field(default_factory=list)
    ticket: Ticket
    constraints: Dict[str, str] = Field(default_factory=dict)


TASKS: Dict[str, TaskDefinition] = {
    "easy_password_reset": TaskDefinition(
        task_id="easy_password_reset",
        level="easy",
        title="Password Reset Guidance",
        objective=(
            "Classify account issue, set appropriate priority, and send a clear reset reply "
            "that references a reset link and 2FA reminder before resolving."
        ),
        max_steps=6,
        required_actions=["set_category", "set_priority", "compose_reply", "resolve_ticket"],
        ticket=Ticket(
            ticket_id="T-1001",
            customer_tier="standard",
            subject="Locked out after phone upgrade",
            description=(
                "I switched phones and now cannot login. Password reset email worked once but "
                "I still fail MFA. Need guidance quickly."
            ),
            sla_minutes=240,
            required_keywords=["reset", "2fa"],
        ),
        constraints={
            "expected_category": "account",
            "expected_priority": "medium",
            "forbid_refund": "true",
        },
    ),
    "medium_billing_duplicate": TaskDefinition(
        task_id="medium_billing_duplicate",
        level="medium",
        title="Duplicate Charge Resolution",
        objective=(
            "Handle a duplicate billing complaint by setting high priority, classifying billing, "
            "adding an internal note, issuing an exact refund, replying with timeline, and resolving."
        ),
        max_steps=10,
        required_actions=[
            "set_priority",
            "set_category",
            "add_internal_note",
            "offer_refund",
            "compose_reply",
            "resolve_ticket",
        ],
        ticket=Ticket(
            ticket_id="T-2024",
            customer_tier="pro",
            subject="Charged twice for one order",
            description=(
                "Card statement shows two charges for invoice INV-8842, each $49.99. "
                "I need correction and confirmation today."
            ),
            order_amount=49.99,
            sla_minutes=120,
            required_keywords=["refund", "24 hours", "apolog"],
        ),
        constraints={
            "expected_category": "billing",
            "expected_priority": "high",
            "required_refund": "49.99",
        },
    ),
    "hard_enterprise_outage": TaskDefinition(
        task_id="hard_enterprise_outage",
        level="hard",
        title="Enterprise Outage Mitigation",
        objective=(
            "For an enterprise outage under strict SLA, triage as urgent technical, escalate to L2, "
            "capture impact details internally, request diagnostics, send mitigation guidance with ETA, "
            "and resolve handoff cleanly."
        ),
        max_steps=14,
        required_actions=[
            "set_priority",
            "set_category",
            "escalate_to_team",
            "add_internal_note",
            "request_missing_info",
            "compose_reply",
            "resolve_ticket",
        ],
        ticket=Ticket(
            ticket_id="T-9007",
            customer_tier="enterprise",
            subject="EU region API outage impacting checkout",
            description=(
                "Checkout API in EU-west is timing out for 37% of requests since 08:40 UTC. "
                "Need immediate mitigation and ETA for exec update in 30 minutes."
            ),
            order_amount=0.0,
            sla_minutes=30,
            required_keywords=["mitigation", "eta", "workaround"],
        ),
        constraints={
            "expected_category": "technical",
            "expected_priority": "urgent",
            "expected_team": "tech_l2",
        },
    ),
}


def get_task(task_id: str) -> TaskDefinition:
    if task_id not in TASKS:
        raise KeyError(f"Unknown task_id: {task_id}")
    return TASKS[task_id]
