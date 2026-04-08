from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from env.environment import CustomerSupportEnv
from env.models import Action


class ResetRequest(BaseModel):
    task_id: str = "easy_password_reset"


class StepRequest(BaseModel):
    action: Action


app = FastAPI(title="OpenEnv Customer Support", version="1.0.0")
ENV = CustomerSupportEnv()


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "service": "customer-support-sla-openenv",
        "status": "ready",
        "endpoints": ["/health", "/reset", "/step", "/state"],
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/reset")
def reset(payload: ResetRequest) -> Dict[str, Any]:
    try:
        obs = ENV.reset(task_id=payload.task_id)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "observation": obs.model_dump(),
        "reward": 0.0,
        "done": False,
        "info": {"task_id": payload.task_id},
    }


@app.post("/step")
def step(payload: StepRequest) -> Dict[str, Any]:
    try:
        obs, reward, done, info = ENV.step(payload.action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
def state() -> Dict[str, Any]:
    try:
        return ENV.state().model_dump()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
