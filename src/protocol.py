from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .events import Event
from .models import AssignmentDecision, EvaluationResult, GlobalKnowledge, LocalKnowledge
from .world import RideRequest, Vehicle, WorldManager

MonitorOperation = Literal[
    "record_vehicle",
    "record_vehicle_available",
    "record_vehicle_position",
    "reset",
]

DispatchMonitorOperation = Literal[
    "record_request",
    "record_time",
    "remove_request",
    "reset",
]

SlaveCommandKind = Literal["monitor", "execute", "shutdown"]


@dataclass(frozen=True)
class KnowledgeSnapshot:
    vehicles: dict[str, dict[str, Any]]
    open_requests: dict[str, dict[str, Any]]
    current_time: int
    move_sequence: int

    @classmethod
    def from_global(cls, knowledge: GlobalKnowledge) -> "KnowledgeSnapshot":
        return cls(
            vehicles={
                vehicle_id: {
                    "position_id": vehicle.position.id if vehicle.position else None,
                    "max_capacity": vehicle.max_capacity,
                    "properties": dict(vehicle.properties),
                    "is_busy": vehicle.is_busy,
                }
                for vehicle_id, vehicle in knowledge.vehicles.items()
            },
            open_requests={
                request_id: {
                    "earliest_service_time": request.earliest_service_time,
                    "latest_service_time": request.latest_service_time,
                    "start_intersection_id": request.start_intersection.id,
                    "end_intersection_id": request.end_intersection.id,
                    "number_of_customers": request.number_of_customers,
                    "remaining_customers": request.remaining_customers,
                }
                for request_id, request in knowledge.open_requests.items()
            },
            current_time=knowledge.current_time,
            move_sequence=knowledge.move_sequence,
        )


@dataclass
class KnowledgeUpdate:
    set_vehicles: dict[str, dict[str, Any]] = field(default_factory=dict)
    set_vehicle_busy: dict[str, bool] = field(default_factory=dict)
    set_vehicle_position: dict[str, int | None] = field(default_factory=dict)
    set_open_requests: dict[str, dict[str, Any]] = field(default_factory=dict)
    remove_open_requests: list[str] = field(default_factory=list)
    set_current_time: int | None = None
    reset: bool = False
    clear_last_objectives: bool = False
    set_move_sequence: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "set_vehicles": self.set_vehicles,
            "set_vehicle_busy": self.set_vehicle_busy,
            "set_vehicle_position": self.set_vehicle_position,
            "set_open_requests": self.set_open_requests,
            "remove_open_requests": self.remove_open_requests,
            "set_current_time": self.set_current_time,
            "reset": self.reset,
            "clear_last_objectives": self.clear_last_objectives,
            "set_move_sequence": self.set_move_sequence,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "KnowledgeUpdate":
        return cls(
            set_vehicles=dict(payload.get("set_vehicles") or {}),
            set_vehicle_busy=dict(payload.get("set_vehicle_busy") or {}),
            set_vehicle_position=dict(payload.get("set_vehicle_position") or {}),
            set_open_requests=dict(payload.get("set_open_requests") or {}),
            remove_open_requests=list(payload.get("remove_open_requests") or []),
            set_current_time=payload.get("set_current_time"),
            reset=bool(payload.get("reset")),
            clear_last_objectives=bool(payload.get("clear_last_objectives")),
            set_move_sequence=payload.get("set_move_sequence"),
        )


def local_snapshot_to_dict(local: LocalKnowledge) -> dict[str, Any]:
    return {
        "vehicle_id": local.vehicle_id,
        "last_monitor_operation": local.last_monitor_operation,
        "last_monitored_event": local.last_monitored_event,
        "last_executed_decisions": list(local.last_executed_decisions),
        "active_route": list(local.active_route),
    }


def local_snapshot_from_dict(payload: dict[str, Any]) -> LocalKnowledge:
    local = LocalKnowledge(vehicle_id=str(payload["vehicle_id"]))
    local.last_monitor_operation = payload.get("last_monitor_operation")
    local.last_monitored_event = payload.get("last_monitored_event")
    local.last_executed_decisions = list(payload.get("last_executed_decisions") or [])
    local.active_route = list(payload.get("active_route") or [])
    return local


def apply_local_snapshot(local: LocalKnowledge, payload: dict[str, Any]) -> None:
    applied = local_snapshot_from_dict(payload)
    local.last_monitor_operation = applied.last_monitor_operation
    local.last_monitored_event = applied.last_monitored_event
    local.last_executed_decisions = applied.last_executed_decisions
    local.active_route = applied.active_route


def snapshot_to_dict(snapshot: KnowledgeSnapshot) -> dict[str, Any]:
    return {
        "vehicles": snapshot.vehicles,
        "open_requests": snapshot.open_requests,
        "current_time": snapshot.current_time,
        "move_sequence": snapshot.move_sequence,
    }


def snapshot_from_dict(payload: dict[str, Any]) -> KnowledgeSnapshot:
    return KnowledgeSnapshot(
        vehicles=dict(payload.get("vehicles") or {}),
        open_requests=dict(payload.get("open_requests") or {}),
        current_time=int(payload.get("current_time", 0) or 0),
        move_sequence=int(payload.get("move_sequence", 0) or 0),
    )


def evaluation_result_to_payload(result: EvaluationResult) -> dict[str, Any]:
    decisions: list[dict[str, Any]] = []
    for decision in result.decisions:
        decisions.append(
            {
                "vehicle_id": decision.vehicle.id,
                "request_id": decision.request.id,
                "served_customers": decision.served_customers,
                "pickup_intersection_id": decision.request.start_intersection.id,
                "destination_intersection_id": decision.request.end_intersection.id,
                "pickup_road_ids": [road.id for road in decision.to_pickup.path.edge_list],
                "destination_road_ids": [road.id for road in decision.to_destination.path.edge_list],
                "travel_time": decision.travel_time,
                "distance": decision.distance,
                "co2_emissions": decision.co2_emissions,
                "distance_cost": decision.distance_cost,
            }
        )
    return {"decisions": decisions}


def decision_to_payload(decision: AssignmentDecision) -> dict[str, Any]:
    return {
        "vehicle_id": decision.vehicle.id,
        "request_id": decision.request.id,
        "served_customers": decision.served_customers,
        "pickup_intersection_id": decision.request.start_intersection.id,
        "destination_intersection_id": decision.request.end_intersection.id,
        "pickup_road_ids": [road.id for road in decision.to_pickup.path.edge_list],
        "destination_road_ids": [road.id for road in decision.to_destination.path.edge_list],
        "travel_time": decision.travel_time,
        "distance": decision.distance,
        "co2_emissions": decision.co2_emissions,
        "distance_cost": decision.distance_cost,
    }


def apply_knowledge_update(
    knowledge: GlobalKnowledge,
    update: KnowledgeUpdate,
    world: WorldManager,
) -> None:
    if update.reset:
        knowledge.vehicles.clear()
        knowledge.open_requests.clear()
        knowledge.current_time = 0
        knowledge.last_objectives = None
        knowledge.cached_chromosomes.clear()
        knowledge.best_chromosome = None
        knowledge.last_reoptimization_time = None
        knowledge.move_sequence = 0
        knowledge.world.on_reset()

    if update.clear_last_objectives:
        knowledge.last_objectives = None

    if update.set_current_time is not None:
        knowledge.current_time = update.set_current_time

    if update.set_move_sequence is not None:
        knowledge.move_sequence = update.set_move_sequence

    for vehicle_id, vehicle_data in update.set_vehicles.items():
        position_id = vehicle_data.get("position_id")
        intersection = world.get_intersection(position_id) if position_id is not None else None
        knowledge.vehicles[vehicle_id] = Vehicle(
            vehicle_id,
            intersection,
            int(vehicle_data["max_capacity"]),
            dict(vehicle_data.get("properties") or {}),
            bool(vehicle_data.get("is_busy", False)),
        )

    for vehicle_id, is_busy in update.set_vehicle_busy.items():
        vehicle = knowledge.vehicles.get(vehicle_id)
        if vehicle is not None:
            vehicle.is_busy = is_busy

    for vehicle_id, position_id in update.set_vehicle_position.items():
        vehicle = knowledge.vehicles.get(vehicle_id)
        if vehicle is None:
            continue
        vehicle.position = world.get_intersection(position_id) if position_id is not None else None

    for request_id, request_data in update.set_open_requests.items():
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
        knowledge.open_requests[request_id] = request

    for request_id in update.remove_open_requests:
        knowledge.open_requests.pop(request_id, None)


def events_from_mappings(mappings: list[dict[str, Any]]) -> list[Event]:
    return [
        Event(
            str(mapping["category"]),
            str(mapping["name"]),
            dict(mapping.get("data") or {}),
            dict(mapping.get("explanations") or {}),
        )
        for mapping in mappings
    ]


def events_to_mappings(events: list[Event]) -> list[dict[str, Any]]:
    return [event.to_mapping() for event in events]
