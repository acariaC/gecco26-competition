from __future__ import annotations

import logging
from typing import Any

from .events import Event
from .models import AssignmentDecision, GlobalKnowledge, LocalKnowledge
from .protocol import (
    KnowledgeSnapshot,
    KnowledgeUpdate,
    MonitorOperation,
    snapshot_from_dict,
)
from .world import RideRequest, Vehicle, WorldManager

LOGGER = logging.getLogger(__name__)


class _SnapshotWorld(WorldManager):


    def __init__(self, intersection_ids: list[int]) -> None:
        super().__init__("http://localhost:8088")
        self.add_intersections([{"id": intersection_id} for intersection_id in intersection_ids])

    def on_reset(self) -> None:
        self.intersections.clear()
        self._adjacency.clear()


def _global_from_snapshot(snapshot: KnowledgeSnapshot, world: WorldManager) -> GlobalKnowledge:
    from .config import EvolutionarySettings

    global_knowledge = GlobalKnowledge(world=world, settings=EvolutionarySettings())
    global_knowledge.current_time = snapshot.current_time
    global_knowledge.move_sequence = snapshot.move_sequence

    for vehicle_id, vehicle_data in snapshot.vehicles.items():
        position_id = vehicle_data.get("position_id")
        intersection = world.get_intersection(position_id) if position_id is not None else None
        global_knowledge.vehicles[vehicle_id] = Vehicle(
            vehicle_id,
            intersection,
            int(vehicle_data["max_capacity"]),
            dict(vehicle_data.get("properties") or {}),
            bool(vehicle_data.get("is_busy", False)),
        )

    for request_id, request_data in snapshot.open_requests.items():
        start = world.get_intersection(request_data["start_intersection_id"])
        end = world.get_intersection(request_data["end_intersection_id"])
        if start is None or end is None:
            continue
        request = RideRequest(
            earliest_service_time=int(request_data["earliest_service_time"]),
            latest_service_time=int(request_data["latest_service_time"]),
            start_intersection=start,
            end_intersection=end,
            id=request_id,
            number_of_customers=int(request_data["number_of_customers"]),
        )
        request.remaining_customers = int(request_data["remaining_customers"])
        global_knowledge.open_requests[request_id] = request

    return global_knowledge


def _snapshot_delta(before: KnowledgeSnapshot, after: GlobalKnowledge) -> KnowledgeUpdate:
    update = KnowledgeUpdate()

    if after.current_time != before.current_time:
        update.set_current_time = after.current_time

    if after.move_sequence != before.move_sequence:
        update.set_move_sequence = after.move_sequence

    for request_id, request in after.open_requests.items():
        before_request = before.open_requests.get(request_id)
        payload = {
            "earliest_service_time": request.earliest_service_time,
            "latest_service_time": request.latest_service_time,
            "start_intersection_id": request.start_intersection.id,
            "end_intersection_id": request.end_intersection.id,
            "number_of_customers": request.number_of_customers,
            "remaining_customers": request.remaining_customers,
        }
        if before_request != payload:
            update.set_open_requests[request_id] = payload

    for request_id in before.open_requests:
        if request_id not in after.open_requests:
            update.remove_open_requests.append(request_id)

    for vehicle_id, vehicle in after.vehicles.items():
        before_vehicle = before.vehicles.get(vehicle_id)
        payload = {
            "position_id": vehicle.position.id if vehicle.position else None,
            "max_capacity": vehicle.max_capacity,
            "properties": dict(vehicle.properties),
            "is_busy": vehicle.is_busy,
        }
        if before_vehicle != payload:
            update.set_vehicles[vehicle_id] = payload
        elif before_vehicle is not None and before_vehicle.get("is_busy") != vehicle.is_busy:
            update.set_vehicle_busy[vehicle_id] = vehicle.is_busy

        before_position = before_vehicle.get("position_id") if before_vehicle else None
        current_position = vehicle.position.id if vehicle.position else None
        if before_position != current_position:
            update.set_vehicle_position[vehicle_id] = current_position

    return update


def process_monitor_command(
    vehicle_id: str,
    snapshot_payload: dict[str, Any],
    local_snapshot_payload: dict[str, Any],
    operation: MonitorOperation,
    event_payload: dict[str, Any],
    intersection_ids: list[int],
) -> dict[str, Any]:
    from .optimizer import TaxiMonitor
    from .protocol import local_snapshot_from_dict, local_snapshot_to_dict

    snapshot = snapshot_from_dict(snapshot_payload)
    world = _SnapshotWorld(intersection_ids)
    global_knowledge = _global_from_snapshot(snapshot, world)
    local_knowledge = local_snapshot_from_dict(local_snapshot_payload)
    event = Event.from_mapping(event_payload)
    monitor = TaxiMonitor(vehicle_id, global_knowledge, local_knowledge)

    if operation == "reset":
        monitor.reset()
        return {
            "update": KnowledgeUpdate().to_dict(),
            "local_snapshot": local_snapshot_to_dict(local_knowledge),
        }

    operation_handler = getattr(monitor, operation)
    operation_handler(event)
    return {
        "update": _snapshot_delta(snapshot, global_knowledge).to_dict(),
        "local_snapshot": local_snapshot_to_dict(local_knowledge),
    }


def process_execute_decision_command(
    vehicle_id: str,
    snapshot_payload: dict[str, Any],
    local_snapshot_payload: dict[str, Any],
    decision_payload: dict[str, Any],
    intersection_ids: list[int],
) -> dict[str, Any]:
    from .optimizer import TaxiExecute
    from .protocol import local_snapshot_from_dict, local_snapshot_to_dict

    snapshot = snapshot_from_dict(snapshot_payload)
    world = _SnapshotWorld(intersection_ids)
    global_knowledge = _global_from_snapshot(snapshot, world)
    local_knowledge = local_snapshot_from_dict(local_snapshot_payload)
    decision = _decision_from_payload(global_knowledge, decision_payload)
    executor = TaxiExecute(vehicle_id, global_knowledge, local_knowledge)

    if decision is None:
        return {
            "events": [],
            "update": KnowledgeUpdate().to_dict(),
            "local_snapshot": local_snapshot_to_dict(local_knowledge),
        }

    events = executor.apply_decision(decision)
    return {
        "events": [event.to_mapping() for event in events],
        "update": _snapshot_delta(snapshot, global_knowledge).to_dict(),
        "local_snapshot": local_snapshot_to_dict(local_knowledge),
    }


def _decision_from_payload(
    global_knowledge: GlobalKnowledge,
    payload: dict[str, Any],
) -> AssignmentDecision | None:
    from .models import AssignmentDecision
    from .world import GraphPath, Road, RouteCandidate

    vehicle = global_knowledge.vehicles.get(str(payload["vehicle_id"]))
    request = global_knowledge.open_requests.get(str(payload["request_id"]))
    if vehicle is None or request is None:
        return None

    pickup_roads = [
        Road(road_id, 0.0, 0.0, {"id": road_id}) for road_id in payload["pickup_road_ids"]
    ]
    destination_roads = [
        Road(road_id, 0.0, 0.0, {"id": road_id}) for road_id in payload["destination_road_ids"]
    ]
    to_pickup = RouteCandidate(
        path=GraphPath(pickup_roads, float(payload["travel_time"])),
        distance=float(payload["distance"]),
        travel_time=float(payload["travel_time"]),
        co2_emissions=float(payload["co2_emissions"]),
        distance_cost=float(payload["distance_cost"]),
    )
    to_destination = RouteCandidate(
        path=GraphPath(destination_roads, float(payload["travel_time"])),
        distance=float(payload["distance"]),
        travel_time=float(payload["travel_time"]),
        co2_emissions=float(payload["co2_emissions"]),
        distance_cost=float(payload["distance_cost"]),
    )
    return AssignmentDecision(
        vehicle=vehicle,
        request=request,
        to_pickup=to_pickup,
        to_destination=to_destination,
        served_customers=int(payload["served_customers"]),
        travel_time=float(payload["travel_time"]),
        distance=float(payload["distance"]),
        co2_emissions=float(payload["co2_emissions"]),
        distance_cost=float(payload["distance_cost"]),
    )


__all__ = [
    "process_execute_decision_command",
    "process_monitor_command",
]
