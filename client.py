import os

import requests

BASE_URL = os.getenv("OPENENV_BASE_URL", "http://localhost:8000")


class OpenEnvClient:
    def __init__(self) -> None:
        self._base = BASE_URL.rstrip("/")

    def reset(self) -> dict:
        r = requests.post(f"{self._base}/reset", timeout=30)
        r.raise_for_status()
        return r.json()

    def step(self, action_type: str, value: str) -> dict:
        r = requests.post(
            f"{self._base}/step",
            json={"action_type": action_type, "value": value},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()


if __name__ == "__main__":
    client = OpenEnvClient()
    print(client.reset())
    print(client.step("classify", "billing"))
