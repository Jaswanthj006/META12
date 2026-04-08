import os

from fastapi import FastAPI
import uvicorn

from customer_support_env import CustomerSupportEnv

app = FastAPI()
env = CustomerSupportEnv()


@app.post("/reset")
def reset() -> dict:
    obs = env.reset()
    return obs.dict()


@app.post("/step")
def step(action: dict) -> dict:
    observation, reward, done, info = env.step(action)
    return {
        "observation": observation.dict(),
        "reward": reward,
        "done": done,
        "info": info,
    }


def main() -> None:
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
