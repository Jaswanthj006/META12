"""
OpenEnv inference: HTTP client + rule-based agent for deployed Customer Support env.
"""
from __future__ import annotations

import json
import os
from typing import Any
from urllib import error as urllib_error
from urllib import request


API_BASE_URL = os.getenv("API_BASE_URL", "").rstrip("/")
MODEL_NAME = os.getenv("MODEL_NAME", "rule-based")
HF_TOKEN = os.getenv("HF_TOKEN", "")

TASK_NAME = "customer_support_triage"
ENV_NAME = "customer_support_env"


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _bool_lower(x: bool) -> str:
    return str(x).lower()


def _rule_classify(ticket_text: str) -> str:
    t = ticket_text.lower()
    delivery_hits = sum(
        1
        for w in (
            "package",
            "delivered",
            "arrived",
            "order says",
            "never received",
            "where is my",
        )
        if w in t
    )
    billing_hits = sum(
        1 for w in ("payment", "charged", "invoice", "refund") if w in t
    )
    if delivery_hits >= 1 and billing_hits == 0:
        return "delivery"
    if billing_hits >= 1:
        return "billing"
    return "technical"


def _rule_priority(ticket_text: str) -> str:
    t = ticket_text.lower()
    if "crash" in t or "twice" in t:
        return "high"
    if "says delivered" in t or ("delivered" in t and "never" in t):
        return "high"
    if "invoice" in t and "copy" in t:
        return "low"
    if "8 days" in t or "password" in t:
        return "medium"
    return "medium"


def _rule_action(ticket_text: str) -> str:
    t = ticket_text.lower()
    if "crash" in t or ("password" in t and "reset" in t):
        return "escalate"
    if "says delivered" in t or ("delivered" in t and "never" in t and "received" in t):
        return "resolve"
    return "respond"


def main() -> None:
    base = API_BASE_URL or "http://127.0.0.1:7860"
    base = base.rstrip("/")

    print(
        f"[START] task={TASK_NAME} env={ENV_NAME} model={MODEL_NAME}",
        flush=True,
    )

    rewards: list[float] = []
    step_num = 0
    done = False
    last_obs: dict[str, Any] = {}
    any_error = False

    try:
        last_obs = _post_json(f"{base}/reset", {})
    except (urllib_error.URLError, urllib_error.HTTPError, OSError, ValueError) as e:
        score = 0.0
        print(
            f"[END] success=false steps=0 score={score:.2f} rewards=",
            flush=True,
        )
        return

    ticket_text = str(last_obs.get("ticket_text", ""))

    planned = [
        ("classify", _rule_classify(ticket_text)),
        ("set_priority", _rule_priority(ticket_text)),
        ("take_action", _rule_action(ticket_text)),
    ]

    for action_type, value in planned:
        step_num += 1
        payload = {"action_type": action_type, "value": value}
        action_label = f"{action_type}:{value}"
        err_out = "null"
        reward = 0.0

        try:
            result = _post_json(f"{base}/step", payload)
            last_obs = result.get("observation", last_obs)
            reward = float(result.get("reward", 0.0))
            done = bool(result.get("done", False))
            info = result.get("info") or {}
            if isinstance(info, dict) and info.get("error"):
                err_out = str(info["error"])
                any_error = True
        except (urllib_error.URLError, urllib_error.HTTPError, OSError, ValueError) as e:
            err_out = str(e)
            done = False
            any_error = True

        rewards.append(reward)
        print(
            f"[STEP] step={step_num} action={action_label} reward={reward:.2f} "
            f"done={_bool_lower(done)} error={err_out}",
            flush=True,
        )

        if done:
            break

    # Score: normalized total in [0, 1] (sum of step rewards capped)
    total = sum(rewards)
    score = max(0.0, min(1.0, total))
    success = bool(done and not any_error)

    rewards_fmt = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={_bool_lower(success)} steps={step_num} "
        f"score={score:.2f} rewards={rewards_fmt}",
        flush=True,
    )


if __name__ == "__main__":
    main()
