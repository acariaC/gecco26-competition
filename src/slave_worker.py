from __future__ import annotations

import logging
from typing import Any

from .slave_ops import process_execute_decision_command, process_monitor_command

LOGGER = logging.getLogger(__name__)


def run_taxi_slave(vehicle_id: str, command_queue: Any, response_queue: Any) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [taxi-%(process)s] %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info("Taxi slave %s started.", vehicle_id)

    while True:
        command = command_queue.get()
        if command is None:
            break

        kind = command.get("kind")
        if kind == "shutdown":
            logger.info("Taxi slave %s shutting down.", vehicle_id)
            break

        request_id = command.get("request_id")
        try:
            if kind == "monitor":
                result = process_monitor_command(
                    vehicle_id,
                    command["snapshot"],
                    command["local_snapshot"],
                    command["operation"],
                    command["event"],
                    list(command.get("intersection_ids") or []),
                )
                response_queue.put(
                    {
                        "request_id": request_id,
                        "vehicle_id": vehicle_id,
                        "kind": "monitor",
                        "update": result["update"],
                        "local_snapshot": result["local_snapshot"],
                    }
                )
            elif kind == "execute_decision":
                result = process_execute_decision_command(
                    vehicle_id,
                    command["snapshot"],
                    command["local_snapshot"],
                    command["decision"],
                    list(command.get("intersection_ids") or []),
                )
                response_queue.put(
                    {
                        "request_id": request_id,
                        "vehicle_id": vehicle_id,
                        "kind": "execute_decision",
                        "events": result["events"],
                        "update": result["update"],
                        "local_snapshot": result["local_snapshot"],
                    }
                )
            else:
                response_queue.put(
                    {
                        "request_id": request_id,
                        "vehicle_id": vehicle_id,
                        "kind": "error",
                        "message": f"Unknown command kind: {kind}",
                    }
                )
        except Exception as error:
            logger.exception("Taxi slave %s failed to process %s command.", vehicle_id, kind)
            response_queue.put(
                {
                    "request_id": request_id,
                    "vehicle_id": vehicle_id,
                    "kind": "error",
                    "message": str(error),
                }
            )

    response_queue.put({"kind": "stopped", "vehicle_id": vehicle_id})


                            
run_slave = run_taxi_slave
