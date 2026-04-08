from typing import Optional

from pydantic import BaseModel


class Observation(BaseModel):
    ticket_id: int
    ticket_text: str
    category: Optional[str] = None
    priority: Optional[str] = None
    status: str


class Action(BaseModel):
    action_type: str
    value: str


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict
