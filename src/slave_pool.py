from __future__ import annotations

import multiprocessing as mp
import uuid
from typing import Any

from .events import Event
from .models import AssignmentDecision, GlobalKnowledge, LocalKnowledge
from .protocol import (
    KnowledgeSnapshot,
    KnowledgeUpdate,
    MonitorOperation,
    apply_knowledge_update,
    apply_local_snapshot,
    decision_to_payload,
    events_from_mappings,
    local_snapshot_to_dict,
    snapshot_to_dict,
)
from .slave_worker import run_taxi_slave
from .world import WorldManager


class TaxiProcessPool:


    def __init__(self) -> None:
        self._local_knowledge: dict[str, LocalKnowledge] = {}
        self._context = mp.get_context("spawn")
        self._command_queues: dict[str, Any] = {}
        self._response_queue = self._context.Queue()
        self._processes: dict[str, mp.Process] = {}

    def has_taxi(self, vehicle_id: str) -> bool:
        return vehicle_id in self._command_queues

    @property
    def is_alive(self) -> bool:
        return any(process.is_alive() for process in self._processes.values())

    def local_knowledge_for(self, vehicle_id: str) -> LocalKnowledge | None:
        return self._local_knowledge.get(vehicle_id)

    def reset_local_knowledge(self) -> None:
        for local in self._local_knowledge.values():
            local.reset()

    def register(self, vehicle_id: str) -> None:
        if vehicle_id in self._command_queues:
            return

        self._local_knowledge[vehicle_id] = LocalKnowledge(vehicle_id=vehicle_id)
        command_queue = self._context.Queue()
        process = self._context.Process(
            target=run_taxi_slave,
            args=(vehicle_id, command_queue, self._response_queue),
            name=f"taxi-slave-{vehicle_id}",
            daemon=True,
        )
        process.start()
        self._command_queues[vehicle_id] = command_queue
        self._processes[vehicle_id] = process

    def monitor(
        self,
        vehicle_id: str,
        global_knowledge: GlobalKnowledge,
        world: WorldManager,
        operation: MonitorOperation,
        event: Event,
    ) -> None:
        local = self._local_knowledge[vehicle_id]
        response = self._dispatch(
            vehicle_id,
            {
                "kind": "monitor",
                "operation": operation,
                "event": event.to_mapping(),
                "snapshot": snapshot_to_dict(KnowledgeSnapshot.from_global(global_knowledge)),
                "local_snapshot": local_snapshot_to_dict(local),
                "intersection_ids": sorted(world.intersections),
            },
        )
        apply_knowledge_update(
            global_knowledge,
            KnowledgeUpdate.from_dict(response["update"]),
            world,
        )
        apply_local_snapshot(local, response["local_snapshot"])

    def execute_decision(
        self,
        vehicle_id: str,
        global_knowledge: GlobalKnowledge,
        world: WorldManager,
        decision: AssignmentDecision,
    ) -> list[Event]:
        local = self._local_knowledge[vehicle_id]
        response = self._dispatch(
            vehicle_id,
            {
                "kind": "execute_decision",
                "decision": decision_to_payload(decision),
                "snapshot": snapshot_to_dict(KnowledgeSnapshot.from_global(global_knowledge)),
                "local_snapshot": local_snapshot_to_dict(local),
                "intersection_ids": sorted(world.intersections),
            },
        )
        apply_knowledge_update(
            global_knowledge,
            KnowledgeUpdate.from_dict(response["update"]),
            world,
        )
        apply_local_snapshot(local, response["local_snapshot"])
        return events_from_mappings(response.get("events") or [])

    def shutdown(self) -> None:
        for vehicle_id, command_queue in self._command_queues.items():
            command_queue.put({"kind": "shutdown", "request_id": f"shutdown-{vehicle_id}"})

        for process in self._processes.values():
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)

        self._command_queues.clear()
        self._processes.clear()
        self._local_knowledge.clear()

    def _dispatch(self, vehicle_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        request_id = uuid.uuid4().hex
        payload = dict(payload)
        payload["request_id"] = request_id
        self._command_queues[vehicle_id].put(payload)

        while True:
            response = self._response_queue.get()
            if response.get("request_id") != request_id:
                continue
            if response.get("kind") == "error":
                raise RuntimeError(response.get("message", "Taxi slave command failed."))
            return response


                            
ProcessSlavePool = TaxiProcessPool
