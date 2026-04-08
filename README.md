---
title: Customer Support Env
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
---

# Customer Support Ticket Triage OpenEnv Environment

This project is a small OpenEnv environment that simulates customer support ticket handling.  
An AI agent gets one ticket at a time and has to process it like a support teammate would: figure out what kind of issue it is, how urgent it is, and what to do next.

## What problem this solves

Support teams deal with repetitive ticket triage every day. Doing this manually for every single ticket takes time, and important tickets can get delayed.

This environment gives a clean way to test automation logic before using it in real workflows. You can evaluate whether an agent is making useful decisions without connecting to real customer systems.

## How the environment works

Each episode starts with a support ticket.  
The agent then works through three decisions:

- classify the ticket
- assign a priority
- choose an action

The ticket text is visible in the observation, but ground-truth labels stay internal to the environment.

## Tasks

The environment has three task levels:

- **Classification**: identify the issue type (`billing`, `technical`, `delivery`)
- **Priority**: decide urgency (`low`, `medium`, `high`)
- **Action**: choose next step (`respond`, `escalate`, `resolve`)

This setup mirrors a real support flow: understand the issue first, then urgency, then execution.

## Reward system

I used weighted rewards so the agent is not judged on a single final action only:

- classification: `0.4`
- priority: `0.3`
- action: `0.3`

Classification gets slightly higher weight because bad routing early in the flow usually causes bigger downstream mistakes.

There is also partial scoring (`0.5`) for close answers in some tasks. That helps measure progress instead of treating everything as all-or-nothing.

A small penalty is applied when a prediction is completely wrong (`score == 0.0`). This discourages careless guesses.  
There is also a bonus for completing all three parts correctly, with final reward safely capped in `[0.0, 1.0]`.

## How to run

### 1) Validate

```bash
openenv validate
```

### 2) Run server

```bash
server
```

or:

```bash
python -m server.app
```

### 3) Deploy

Use your normal OpenEnv deployment flow (after validation passes):

```bash
openenv deploy
```

## API usage

The server exposes two endpoints:

- `POST /reset`: starts a new episode and returns initial observation
- `POST /step`: applies one action and returns observation, reward, done, info

Example action payload for `/step`:

```json
{
  "action_type": "classify",
  "value": "billing"
}
```

## Inference script

`inference.py` runs an LLM-driven loop against the environment API.  
It prints logs in this format:

- `[START] ...`
- `[STEP] ...`
- `[END] ...`

That makes it easy to read trajectories quickly during testing.

## Final note

This project is a practical testbed for checking how well LLM agents handle real support-style workflows end to end, not just isolated prompts.
