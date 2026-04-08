from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union, cast

from pydantic import BaseModel


class Observation(BaseModel):
    ticket_id: int
    ticket_text: str
    category: Optional[str] = None
    priority: Optional[str] = None
    status: Literal["open", "in_progress", "resolved"] = "open"


class Action(BaseModel):
    action_type: Literal["classify", "set_priority", "take_action"]
    value: str


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]


class CustomerSupportEnv:
    """
    Deterministic environment skeleton for customer support ticket handling.

    Notes:
    - No reward logic is implemented yet (placeholder reward = 0.0).
    - No grading/correctness validation is implemented yet.
    - No randomness is used anywhere.
    """

    def __init__(self) -> None:
        self.task_definitions: Dict[str, Dict[str, Any]] = {
            "easy": {
                "action_type": "classify",
                "valid_values": ["billing", "technical", "delivery"],
            },
            "medium": {
                "action_type": "set_priority",
                "valid_values": ["low", "medium", "high"],
            },
            "hard": {
                "action_type": "take_action",
                "valid_values": ["respond", "escalate", "resolve"],
            },
        }

        self._tickets: List[Dict[str, Any]] = [
            {
                "ticket_id": 1,
                "ticket_text": "My payment was charged twice for the same order.",
                "category": "billing",
                "priority": "high",
                "correct_action": "respond",
            },
            {
                "ticket_id": 2,
                "ticket_text": "The mobile app crashes every time I try to log in.",
                "category": "technical",
                "priority": "high",
                "correct_action": "escalate",
            },
            {
                "ticket_id": 3,
                "ticket_text": "Where is my package? It has not arrived after 8 days.",
                "category": "delivery",
                "priority": "medium",
                "correct_action": "respond",
            },
            {
                "ticket_id": 4,
                "ticket_text": "I need an invoice copy for last month.",
                "category": "billing",
                "priority": "low",
                "correct_action": "respond",
            },
            {
                "ticket_id": 5,
                "ticket_text": "I cannot reset my password using the email link.",
                "category": "technical",
                "priority": "medium",
                "correct_action": "escalate",
            },
            {
                "ticket_id": 6,
                "ticket_text": "My order says delivered, but I never received it.",
                "category": "delivery",
                "priority": "high",
                "correct_action": "resolve",
            },
        ]

        self.current_index: int = 0
        self.current_ticket: Optional[Dict[str, Any]] = None
        self.step_count: int = 0
        self._observation: Optional[Observation] = None
        self.progress: Dict[str, bool] = {
            "classification_done": False,
            "priority_done": False,
            "action_done": False,
        }

    def grade_classification(self, predicted: str, actual: str) -> float:
        if predicted == actual:
            return 1.0
        return 0.0

    def grade_priority(self, predicted: str, actual: str) -> float:
        if predicted == actual:
            return 1.0

        close_pairs = {
            frozenset(("medium", "high")),
            frozenset(("medium", "low")),
        }
        if frozenset((predicted, actual)) in close_pairs:
            return 0.5
        return 0.0

    def grade_action(self, predicted: str, actual: str) -> float:
        if predicted == actual:
            return 1.0

        acceptable_alternatives = {
            frozenset(("respond", "resolve")),
        }
        if frozenset((predicted, actual)) in acceptable_alternatives:
            return 0.5
        return 0.0

    def reset(self) -> Observation:
        # Fixed dataset order (deterministic; no shuffling).
        self.current_index = 0
        self.step_count = 0
        self.progress = {
            "classification_done": False,
            "priority_done": False,
            "action_done": False,
        }
        self.current_ticket = self._tickets[self.current_index]
        self._observation = Observation(
            ticket_id=self.current_ticket["ticket_id"],
            ticket_text=self.current_ticket["ticket_text"],
            category=None,
            priority=None,
            status="open",
        )
        return self._observation

    def step(
        self, action: Union[Action, Dict[str, Any]]
    ) -> tuple[Observation, float, bool, Dict[str, Any]]:
        if self._observation is None or self.current_ticket is None:
            raise RuntimeError("Environment must be reset() before step().")

        if isinstance(action, dict):
            raw_type = action.get("action_type")
            raw_value = str(action.get("value", ""))
        else:
            raw_type = action.action_type
            raw_value = action.value

        if raw_type not in ("classify", "set_priority", "take_action"):
            return self._observation, 0.0, False, {"error": "invalid_action"}

        action = Action(
            action_type=cast(
                Literal["classify", "set_priority", "take_action"], raw_type
            ),
            value=raw_value,
        )

        self.step_count += 1
        score = 0.0
        reward = 0.0
        expected: str = ""

        if action.action_type == "classify":
            self._observation.category = action.value
            expected = str(self.current_ticket["category"])
            score = self.grade_classification(
                predicted=action.value,
                actual=self.current_ticket["category"],
            )
            reward = score * 0.4
            if score == 0.0:
                reward = max(0.0, reward - 0.1)
            if score >= 0.5:
                self.progress["classification_done"] = True
        elif action.action_type == "set_priority":
            self._observation.priority = action.value
            expected = str(self.current_ticket["priority"])
            score = self.grade_priority(
                predicted=action.value,
                actual=self.current_ticket["priority"],
            )
            reward = score * 0.3
            if score == 0.0:
                reward = max(0.0, reward - 0.1)
            if score >= 0.5:
                self.progress["priority_done"] = True
        elif action.action_type == "take_action":
            if action.value == "respond":
                self._observation.status = "in_progress"
            elif action.value == "resolve":
                self._observation.status = "resolved"
            elif action.value == "escalate":
                self._observation.status = "in_progress"
            expected = str(self.current_ticket["correct_action"])
            score = self.grade_action(
                predicted=action.value,
                actual=self.current_ticket["correct_action"],
            )
            reward = score * 0.3
            if score == 0.0:
                reward = max(0.0, reward - 0.1)
            if score >= 0.5:
                self.progress["action_done"] = True

        bonus = 0.2 if all(self.progress.values()) else 0.0
        total_reward = reward + bonus
        total_reward = min(total_reward, 1.0)
        reward = total_reward
        reward = max(0.0, min(reward, 1.0))

        done = action.action_type == "take_action" or self.step_count >= 5
        info = {
            "task_type": action.action_type,
            "score": score,
            "predicted": action.value,
            "expected": expected,
        }
        return self._observation, reward, done, info

    def state(self) -> Dict[str, int | Dict[str, bool]]:
        current_ticket_id = 0
        if self.current_ticket is not None:
            current_ticket_id = int(self.current_ticket["ticket_id"])
        return {
            "current_ticket": current_ticket_id,
            "step_count": self.step_count,
            "progress": self.progress,
        }

