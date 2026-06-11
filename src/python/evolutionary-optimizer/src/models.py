from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from .config import EvolutionarySettings
from .events import Event
from .world import RideRequest, RouteCandidate, Vehicle, WorldManager

if TYPE_CHECKING:
    from .request_validator import RequestValidator


@dataclass(frozen=True)
class AssignmentCandidate:
    vehicle: Vehicle
    request: RideRequest
    to_pickup: RouteCandidate
    to_destination: RouteCandidate
    travel_time: float
    distance: float
    co2_emissions: float
    distance_cost: float


@dataclass(frozen=True)
class AssignmentChoiceKey:


    request_id: str
    pickup_road_ids: tuple[int, ...]
    destination_road_ids: tuple[int, ...]

    @classmethod
    def from_candidate(cls, candidate: AssignmentCandidate) -> "AssignmentChoiceKey":
        return cls(
            request_id=candidate.request.id,
            pickup_road_ids=candidate.to_pickup.road_ids,
            destination_road_ids=candidate.to_destination.road_ids,
        )


@dataclass(frozen=True)
class CachedChromosome:
    chromosome: np.ndarray
    vehicle_ids: tuple[str, ...]
    choice_keys_by_vehicle: tuple[tuple[AssignmentChoiceKey, ...], ...]

    @classmethod
    def from_snapshot(
        cls,
        chromosome: np.ndarray,
        snapshot: "AssignmentProblemSnapshot",
    ) -> "CachedChromosome":
        return cls(
            chromosome=np.asarray(chromosome, dtype=int).copy(),
            vehicle_ids=snapshot.vehicle_ids,
            choice_keys_by_vehicle=snapshot.choice_keys_by_vehicle,
        )


@dataclass(frozen=True)
class AssignmentProblemSnapshot:


    vehicles: list[Vehicle]
    requests: list[RideRequest]
    choices_by_vehicle: list[list[AssignmentCandidate]]
    current_time: int

    @property
    def empty(self) -> bool:
        return not self.vehicles or not self.requests or not any(self.choices_by_vehicle)

    @property
    def chromosome_length(self) -> int:
        return len(self.vehicles)

    @property
    def upper_bounds(self) -> np.ndarray:
        return np.array([len(choices) for choices in self.choices_by_vehicle], dtype=int)

    @property
    def vehicle_ids(self) -> tuple[str, ...]:
        return tuple(vehicle.id for vehicle in self.vehicles)

    @property
    def request_ids(self) -> tuple[str, ...]:
        return tuple(request.id for request in self.requests)

    @property
    def choice_keys_by_vehicle(self) -> tuple[tuple[AssignmentChoiceKey, ...], ...]:
        return tuple(
            tuple(AssignmentChoiceKey.from_candidate(candidate) for candidate in choices)
            for choices in self.choices_by_vehicle
        )

    def repair_chromosome(self, chromosome: np.ndarray) -> np.ndarray:
        repaired = np.zeros(self.chromosome_length, dtype=int)
        raw = np.asarray(chromosome, dtype=int).flatten()
        for index in range(min(len(raw), self.chromosome_length)):
            repaired[index] = int(np.clip(raw[index], 0, self.upper_bounds[index]))
        return repaired

    def repair_cached_chromosome(self, cached: CachedChromosome) -> np.ndarray:
        repaired = np.zeros(self.chromosome_length, dtype=int)
        vehicle_index_by_id = {vehicle_id: index for index, vehicle_id in enumerate(self.vehicle_ids)}
        current_choice_keys = self.choice_keys_by_vehicle

        for source_index, raw_gene in enumerate(cached.chromosome.astype(int).tolist()):
            if source_index >= len(cached.vehicle_ids):
                break
            if raw_gene <= 0:
                continue

            target_vehicle_index = vehicle_index_by_id.get(cached.vehicle_ids[source_index])
            if target_vehicle_index is None:
                continue

            source_choice_index = raw_gene - 1
            if source_index >= len(cached.choice_keys_by_vehicle):
                continue
            source_choices = cached.choice_keys_by_vehicle[source_index]
            if source_choice_index >= len(source_choices):
                continue

            try:
                target_choice_index = current_choice_keys[target_vehicle_index].index(
                    source_choices[source_choice_index]
                )
            except ValueError:
                continue
            repaired[target_vehicle_index] = target_choice_index + 1

        return repaired


@dataclass(frozen=True)
class AssignmentDecision:
    vehicle: Vehicle
    request: RideRequest
    to_pickup: RouteCandidate
    to_destination: RouteCandidate
    served_customers: int
    travel_time: float
    distance: float
    co2_emissions: float
    distance_cost: float


@dataclass(frozen=True)
class EvaluationResult:
    decisions: list[AssignmentDecision]
    unserved_customers: int
    total_travel_time: float
    total_distance: float
    total_co2_emissions: float
    total_distance_cost: float
    total_late_service_time: float
    duplicate_assignments: int
    served_customers: int
    deadline_violations: int = 0

    @property
    def objectives(self) -> tuple[float, float, float, float]:
        return (
            float(self.unserved_customers),
            float(self.total_travel_time + self.total_late_service_time),
            float(self.total_distance_cost),
            float(self.total_co2_emissions),
        )


@dataclass
class GlobalKnowledge:


    world: WorldManager
    settings: EvolutionarySettings
    move_sequence: int = 0
    vehicles: dict[str, Vehicle] = field(default_factory=dict)
    open_requests: dict[str, RideRequest] = field(default_factory=dict)
    assigned_request_ids: set[str] = field(default_factory=set)
    cancelled_request_ids: set[str] = field(default_factory=set)
    request_latest_service_times: dict[str, int] = field(default_factory=dict)
    request_active_vehicle: dict[str, str] = field(default_factory=dict)
    picked_up_request_ids: set[str] = field(default_factory=set)
    person_request_ids: dict[str, str] = field(default_factory=dict)
    current_time: int = 0
    last_objectives: tuple[float, float, float, float] | None = None
    cached_chromosomes: list[CachedChromosome] = field(default_factory=list)
    best_chromosome: CachedChromosome | None = None
    last_reoptimization_time: int | None = None
    request_validator: RequestValidator | None = None

    def free_vehicles(self) -> list[Vehicle]:
        return [
            vehicle
            for vehicle in self.vehicles.values()
            if not vehicle.is_busy and vehicle.position is not None
        ]

    def request_is_committed(self, request_id: str) -> bool:
        return request_id in self.assigned_request_ids

    def request_is_cancelled(self, request_id: str) -> bool:
        return request_id in self.cancelled_request_ids

    def request_is_picked_up(self, request_id: str) -> bool:
        return request_id in self.picked_up_request_ids

    def mark_request_picked_up(self, request_id: str) -> None:
        self.picked_up_request_ids.add(request_id)

    def mark_request_assigned(self, request_id: str) -> None:
        self.assigned_request_ids.add(request_id)

    def mark_request_cancelled(self, request_id: str) -> None:
        self.cancelled_request_ids.add(request_id)
        self.open_requests.pop(request_id, None)
        self.assigned_request_ids.discard(request_id)
        self.picked_up_request_ids.discard(request_id)
        self.forget_request_deadline(request_id)

    def remember_request_deadline(self, request_id: str, latest_service_time: int) -> None:
        self.request_latest_service_times[request_id] = latest_service_time

    def forget_request_deadline(self, request_id: str) -> None:
        self.request_latest_service_times.pop(request_id, None)

    def request_latest_service_time(self, request_id: str) -> int | None:
        request = self.open_requests.get(request_id)
        if request is not None:
            return request.latest_service_time
        return self.request_latest_service_times.get(request_id)

    def bind_request_vehicle(self, request_id: str, vehicle_id: str) -> None:
        self.request_active_vehicle[request_id] = vehicle_id

    def unbind_request(self, request_id: str) -> str | None:
        return self.request_active_vehicle.pop(request_id, None)

    def unbind_vehicle(self, vehicle_id: str) -> str | None:
        for request_id, bound_vehicle_id in list(self.request_active_vehicle.items()):
            if bound_vehicle_id == vehicle_id:
                del self.request_active_vehicle[request_id]
                return request_id
        return None

    def record_person_request(self, person_id: str, request_id: str) -> None:
        self.person_request_ids[person_id] = request_id

    def resolve_request_for_person(
        self,
        person_id: str,
        request_id: str | None = None,
    ) -> str | None:
        if request_id is not None:
            self.person_request_ids.pop(person_id, None)
            return request_id
        return self.person_request_ids.pop(person_id, None)

    def next_move_id(self) -> str:
        move_id = f"move-{self.move_sequence}"
        self.move_sequence += 1
        return move_id


@dataclass
class LocalKnowledge:


    vehicle_id: str
    last_monitor_operation: str | None = None
    last_monitored_event: dict[str, Any] | None = None
    last_executed_decisions: list[dict[str, Any]] = field(default_factory=list)
    active_route: list[dict[str, Any]] = field(default_factory=list)

    def record_monitor(self, operation: str, event: Event) -> None:
        self.last_monitor_operation = operation
        self.last_monitored_event = event.to_mapping()

    def record_execution(self, decision: AssignmentDecision) -> None:
        payload = {
            "vehicle_id": decision.vehicle.id,
            "request_id": decision.request.id,
            "served_customers": decision.served_customers,
            "travel_time": decision.travel_time,
            "distance": decision.distance,
            "co2_emissions": decision.co2_emissions,
            "distance_cost": decision.distance_cost,
            "pickup_road_ids": [road.id for road in decision.to_pickup.path.edge_list],
            "destination_road_ids": [
                road.id for road in decision.to_destination.path.edge_list
            ],
        }
        self.last_executed_decisions = [payload]
        self.active_route = [
            *[{"type": "follow-road", "road-id": road.id} for road in decision.to_pickup.path.edge_list],
            {
                "type": "pick-up-passengers",
                "request-id": decision.request.id,
                "intersection-id": decision.request.start_intersection.id,
                "count": decision.served_customers,
            },
            *[
                {"type": "follow-road", "road-id": road.id}
                for road in decision.to_destination.path.edge_list
            ],
            {
                "type": "drop-off-passengers",
                "request-id": decision.request.id,
                "intersection-id": decision.request.end_intersection.id,
                "count": decision.served_customers,
            },
        ]

    def reset(self) -> None:
        self.last_monitor_operation = None
        self.last_monitored_event = None
        self.last_executed_decisions.clear()
        self.active_route.clear()


Knowledge = GlobalKnowledge
