# Python Evolutionary Optimizer

Standalone Python optimizer for the GECCO'26 taxi dispatch simulator (Track 1).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python -m src.main
```

Default simulator endpoints:

- `STOMP_SERVER_WS=ws://localhost:8088/simulation-websocket`
- `STOMP_SERVER_HTTP=http://localhost:8088`

## Test

```bash
python -m pytest
python -m compileall src
```
