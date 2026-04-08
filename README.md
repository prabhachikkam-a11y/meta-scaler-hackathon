---
title: Customer Support SLA OpenEnv
emoji: "🤖"
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
---

# Customer Support SLA OpenEnv

## Problem Description
This project implements a production-style RL environment that simulates human customer support operations under SLA constraints. The agent must triage, classify, communicate, and resolve support tickets while avoiding invalid operations and inefficient action loops.

## Real-World Motivation
Modern support teams balance customer satisfaction, response-time SLAs, refund controls, escalation policies, and clear communication. This environment models those trade-offs with deterministic grading and dense rewards so policies can be trained and evaluated reliably.

## Environment Interface
The environment class is `CustomerSupportEnv` and exposes:
- `reset(task_id: str) -> Observation`
- `step(action: Action) -> (Observation, reward, done, info)`
- `state() -> State`

Pydantic models are used for `Action`, `Observation`, `State`, and reward breakdown.

## Action Space
Discrete action types with structured payloads:
- `view_ticket`
- `set_priority` with value in `low|medium|high|urgent`
- `set_category` with value in `account|billing|technical|shipping`
- `add_internal_note` with non-empty text
- `request_missing_info` with non-empty text
- `compose_reply` with minimum length constraints
- `offer_refund` with numeric amount, bounded by order amount
- `escalate_to_team` with value in `billing_ops|tech_l2|risk`
- `resolve_ticket`

## Observation Space
Each observation includes:
- Task metadata: `task_id`, objective, step counters
- Ticket snapshot: customer tier, issue details, SLA, amount
- Workspace state: chosen priority/category, notes count, refund, escalation, resolution status
- `last_action_error` for invalid transition feedback
- `allowed_actions`

## Tasks (3 Levels)
1. Easy: `easy_password_reset`
- Objective: account classification, medium priority, reset+2FA reply, resolve.

2. Medium: `medium_billing_duplicate`
- Objective: billing triage with high priority, internal note, exact refund, timeline reply, resolve.

3. Hard: `hard_enterprise_outage`
- Objective: urgent technical outage handling with L2 escalation, impact notes, diagnostics request, mitigation+ETA workaround response, and handoff resolution.

All graders are deterministic and return scores in `[0.0, 1.0]`.

## Reward Logic
Reward is dense and deterministic:
- Positive progress: score delta from grader component improvements
- Penalty per step: discourages unnecessary long trajectories
- Penalty for invalid actions: malformed or policy-violating transitions
- Penalty for loops/no-op repetition: repeated identical actions beyond a threshold
- Completion bonus: proportional to final score on episode end

## Project Structure
- `openenv.yaml`
- `env/models.py`
- `env/tasks.py`
- `env/graders.py`
- `env/reward.py`
- `env/environment.py`
- `inference.py`
- `app.py`
- `Dockerfile`
- `requirements.txt`
- `README.md`

## Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Run API server:
```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```
3. Test endpoint:
```bash
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d "{}"
```

## How to Run Inference
Set environment variables:
- `API_BASE_URL`
- `MODEL_NAME`
- `API_KEY` (validator-injected LiteLLM proxy key)
- `HF_TOKEN`
- `LOCAL_IMAGE_NAME` (optional, only if running a docker-image based env client)

Then run:
```bash
python inference.py
```

Output format is strict structured logs only:
- `[START] ...`
- `[STEP] ...`
- `[END] ...`

## Baseline Scores
Using the deterministic fallback plan in `inference.py`, expected scores are:
- Easy: `1.000`
- Medium: `1.000`
- Hard: `1.000`

## Hugging Face Space Compatibility
- Containerized server via `Dockerfile`
- Exposes HTTP interface on port `7860`
- Supports `/reset`, `/step`, and `/state`
- Starts with no hidden runtime dependencies
