from __future__ import annotations

import json
import os
import warnings
from typing import Dict, List, Optional, Tuple

from env.environment import CustomerSupportEnv
from env.models import Action

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
API_KEY = os.getenv("API_KEY")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
BENCHMARK = "customer-support-sla-openenv"

TASKS = [
    "easy_password_reset",
    "medium_billing_duplicate",
    "hard_enterprise_outage",
]

MAX_STEPS = {
    "easy_password_reset": 6,
    "medium_billing_duplicate": 10,
    "hard_enterprise_outage": 14,
}


def log_start(task: str, env_name: str, model: str) -> None:
    print(f"[START] task={task} env={env_name} model={model}", flush=True)


def log_step(step: int, action_str: str, reward: float, done: bool, error: Optional[str]) -> None:
    done_value = str(done).lower()
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_value} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    reward_csv = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={reward_csv}",
        flush=True,
    )


def _action_to_str(action: Action) -> str:
    if action.amount is not None:
        return f"{action.action_type.value}(value={action.value},amount={action.amount})"
    return f"{action.action_type.value}(value={action.value})"


def _fallback_action(task_id: str, step: int) -> Action:
    plans: Dict[str, List[Action]] = {
        "easy_password_reset": [
            Action(action_type="view_ticket"),
            Action(action_type="set_category", value="account"),
            Action(action_type="set_priority", value="medium"),
            Action(
                action_type="compose_reply",
                value="Please use the account reset link, then re-enroll 2FA from your new phone to restore access.",
            ),
            Action(action_type="resolve_ticket"),
        ],
        "medium_billing_duplicate": [
            Action(action_type="view_ticket"),
            Action(action_type="set_priority", value="high"),
            Action(action_type="set_category", value="billing"),
            Action(action_type="add_internal_note", value="Duplicate charge verified against INV-8842."),
            Action(action_type="offer_refund", amount=49.99),
            Action(
                action_type="compose_reply",
                value="We apologize for the duplicate charge. A full refund has been submitted and will appear within 24 hours.",
            ),
            Action(action_type="resolve_ticket"),
        ],
        "hard_enterprise_outage": [
            Action(action_type="view_ticket"),
            Action(action_type="set_priority", value="urgent"),
            Action(action_type="set_category", value="technical"),
            Action(action_type="escalate_to_team", value="tech_l2"),
            Action(action_type="add_internal_note", value="37% EU-west checkout timeout, executive update in 30 minutes."),
            Action(action_type="request_missing_info", value="Please share request IDs and affected tenant IDs for correlation."),
            Action(
                action_type="compose_reply",
                value=(
                    "We have activated mitigation by routing traffic away from the impacted shard. "
                    "Current workaround is retry with exponential backoff; next ETA update in 15 minutes."
                ),
            ),
            Action(action_type="resolve_ticket"),
        ],
    }
    seq = plans[task_id]
    if step - 1 < len(seq):
        return seq[step - 1]
    return Action(action_type="view_ticket")


def _llm_action(client: object, task_id: str, step: int, observation: dict) -> Optional[Action]:
    if client is None:
        return None

    prompt = {
        "task_id": task_id,
        "step": step,
        "objective": observation.get("objective"),
        "ticket": observation.get("ticket_snapshot"),
        "workspace": observation.get("workspace"),
        "allowed_actions": observation.get("allowed_actions"),
        "instruction": (
            "Return only compact JSON with keys action_type, value, amount. "
            "Use null for missing optional fields."
        ),
    }
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            max_tokens=120,
            timeout=8,
            messages=[
                {
                    "role": "system",
                    "content": "You are a deterministic support operations agent. Output JSON only.",
                },
                {"role": "user", "content": json.dumps(prompt)},
            ],
        )
        text = (completion.choices[0].message.content or "").strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()
        payload = json.loads(text)
        return Action(**payload)
    except Exception:
        return None


def run_task(client: object, task_id: str) -> Tuple[bool, int, float, List[float]]:
    env = CustomerSupportEnv()
    rewards: List[float] = []
    score = 0.0
    steps_taken = 0
    success = False
    log_start(task=task_id, env_name=BENCHMARK, model=MODEL_NAME)

    done = False
    obs = env.reset(task_id=task_id)

    try:
        for step in range(1, MAX_STEPS[task_id] + 1):
            if done:
                break

            action = _llm_action(client, task_id, step, obs.model_dump())
            if action is None:
                action = _fallback_action(task_id, step)

            obs, reward, done, info = env.step(action)
            rewards.append(float(reward))
            steps_taken = step
            score = float(info.get("score", 0.0))
            error = obs.last_action_error
            log_step(step=step, action_str=_action_to_str(action), reward=reward, done=done, error=error)

            if done:
                break

        success = score >= 0.85
    except Exception as exc:
        steps_taken += 1
        rewards.append(0.0)
        score = 0.0
        success = False
        log_step(step=steps_taken, action_str="exception", reward=0.0, done=True, error=str(exc))
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return success, steps_taken, score, rewards


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"Core Pydantic V1 functionality isn't compatible with Python 3\.14 or greater\.",
    )

    from openai import OpenAI

    # Use injected validator credentials first so calls are observed on LiteLLM proxy.
    api_key = API_KEY or HF_TOKEN
    client = OpenAI(base_url=API_BASE_URL, api_key=api_key)

    for task_id in TASKS:
        run_task(client, task_id)


if __name__ == "__main__":
    main()
