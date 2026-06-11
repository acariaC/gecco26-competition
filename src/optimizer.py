from __future__ import annotations

import itertools
import logging
from multiprocessing import Pool
from typing import Any

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem, Problem
from pymoo.core.sampling import Sampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.optimize import minimize
from pymoo.parallelization import StarmapParallelization

from .config import EvolutionarySettings
from .events import Event
from .models import (
    AssignmentCandidate,
    AssignmentChoiceKey,
    AssignmentDecision,
    AssignmentProblemSnapshot,
    CachedChromosome,
    EvaluationResult,
    GlobalKnowledge,
    Knowledge,
    LocalKnowledge,
)
from .world import RideRequest, RouteCandidate, Vehicle, WorldManager

LOGGER = logging.getLogger(__name__)


def _chromosome_key(chromosome: np.ndarray) -> tuple[int, ...]:
    return tuple(int(gene) for gene in np.asarray(chromosome, dtype=int).flatten().tolist())


def _search_space_size(snapshot: AssignmentProblemSnapshot) -> int:
    size = 1
    for upper_bound in snapshot.upper_bounds.tolist():
        size *= int(upper_bound) + 1
    return size


def _enumerate_chromosomes(snapshot: AssignmentProblemSnapshot) -> list[np.ndarray]:
    ranges = [range(int(upper_bound) + 1) for upper_bound in snapshot.upper_bounds.tolist()]
    return [np.array(genes, dtype=int) for genes in itertools.product(*ranges)]


class DispatchMonitor:


    def __init__(self, global_knowledge: GlobalKnowledge) -> None:
        self.global_knowledge = global_knowledge
        self._pending_reject_events: list[Event] = []

    def flush_reject_events(self) -> list[Event]:

        events = self._pending_reject_events[:]
        self._pending_reject_events.clear()
        return events

    def record_request(self, event: Event) -> None:
        request_id = str(event.get("id"))
        world = self.global_knowledge.world
        world.ensure_road_network_loaded()
        start_intersection = world.get_intersection(
            event.get("start-intersection-id")
        )
        end_intersection = world.get_intersection(
            event.get("end-intersection-id")
        )
        if start_intersection is None or end_intersection is None:
            LOGGER.error(
                "Ignoring request %s because start or end intersection cannot be found.",
                request_id,
            )
            return

        if self.global_knowledge.request_is_committed(request_id):
            return
        if self.global_knowledge.request_is_cancelled(request_id):
            return
        if request_id in self.global_knowledge.open_requests:
            return

                                                                       
        if self.global_knowledge.request_validator is None and world.roads:
            from .request_validator import RequestValidator
            validator = RequestValidator(world)
            validator.refresh_dead_ends()
            self.global_knowledge.request_validator = validator

                                                         
        validator = self.global_knowledge.request_validator
        if validator is not None:
            validation = validator.validate_request(
                request_id,
                int(start_intersection.id),
                int(end_intersection.id),
            )
            if not validation.valid:
                self.global_knowledge.mark_request_cancelled(request_id)
                if validation.reject_event is not None:
                    self._pending_reject_events.append(validation.reject_event)
                return

        latest_service_time = int(event.get("latest-service-time", 2**31 - 1) or 2**31 - 1)
        if self.global_knowledge.current_time > latest_service_time:
            self.global_knowledge.mark_request_cancelled(request_id)
            return

        self.global_knowledge.open_requests[request_id] = RideRequest(
            earliest_service_time=int(event.get("earliest-service-time", 0) or 0),
            latest_service_time=latest_service_time,
            start_intersection=start_intersection,
            end_intersection=end_intersection,
            id=request_id,
            number_of_customers=int(event.get("number-of-customers")),
        )
        self.global_knowledge.remember_request_deadline(request_id, latest_service_time)

    def record_time(self, event: Event) -> None:
        self.global_knowledge.current_time = int(
            event.get("time", self.global_knowledge.current_time) or 0
        )

    def remove_request(self, event: Event) -> None:
        request_id = str(event.get("id"))
        self.global_knowledge.open_requests.pop(request_id, None)
        self.global_knowledge.assigned_request_ids.discard(request_id)
        self.global_knowledge.cancelled_request_ids.discard(request_id)
        self.global_knowledge.forget_request_deadline(request_id)
        self.global_knowledge.unbind_request(request_id)

    def cancel_request(self, request_id: str) -> None:
        self.global_knowledge.mark_request_cancelled(request_id)
        self.global_knowledge.unbind_request(request_id)

    def record_person(self, event: Event) -> None:
        person_id = event.get("id")
        request_id = event.get("request-id")
        if person_id is None or request_id is None:
            return
        self.global_knowledge.record_person_request(str(person_id), str(request_id))

    def reset(self) -> None:
        self.global_knowledge.vehicles.clear()
        self.global_knowledge.open_requests.clear()
        self.global_knowledge.assigned_request_ids.clear()
        self.global_knowledge.cancelled_request_ids.clear()
        self.global_knowledge.request_latest_service_times.clear()
        self.global_knowledge.request_active_vehicle.clear()
        self.global_knowledge.picked_up_request_ids.clear()
        self.global_knowledge.person_request_ids.clear()
        self.global_knowledge.current_time = 0
        self.global_knowledge.move_sequence = 0
        self.global_knowledge.last_objectives = None
        self.global_knowledge.cached_chromosomes.clear()
        self.global_knowledge.best_chromosome = None
        self.global_knowledge.last_reoptimization_time = None
        self.global_knowledge.world.on_reset()


class TaxiMonitor:


    def __init__(
        self,
        vehicle_id: str,
        global_knowledge: GlobalKnowledge,
        local_knowledge: LocalKnowledge,
    ) -> None:
        self.vehicle_id = vehicle_id
        self.global_knowledge = global_knowledge
        self.local_knowledge = local_knowledge

    def record_vehicle(self, event: Event) -> None:
        if str(event.get("id")) != self.vehicle_id:
            return

        world = self.global_knowledge.world
        world.ensure_road_network_loaded()
        intersection = world.get_intersection(event.get("intersection-id"))
        properties = dict(event.get("properties") or {})
        max_capacity = int(properties.get("maximum-capacity", 2**31 - 1))
        existing = self.global_knowledge.vehicles.get(self.vehicle_id)
        is_busy = existing.is_busy if existing is not None else False
        last_active_time = existing.last_active_time if existing is not None else 0
        self.global_knowledge.vehicles[self.vehicle_id] = Vehicle(
            self.vehicle_id,
            intersection,
            max_capacity,
            properties,
            is_busy=is_busy,
            last_active_time=last_active_time,
        )
        self.local_knowledge.record_monitor("record_vehicle", event)

    def record_vehicle_available(self, event: Event) -> None:
        if str(event.get("vehicle-id")) != self.vehicle_id:
            return

        vehicle = self.global_knowledge.vehicles.get(self.vehicle_id)
        if vehicle is None:
            LOGGER.warning("Ignoring availability for unknown taxi %s.", self.vehicle_id)
            return

        vehicle.is_busy = False
        self.global_knowledge.unbind_vehicle(self.vehicle_id)
        self.local_knowledge.record_monitor("record_vehicle_available", event)

    def record_vehicle_position(self, event: Event) -> None:
        if str(event.get("vehicle-id")) != self.vehicle_id:
            return

        intersection_id = int(event.get("intersection-id"))
        vehicle = self.global_knowledge.vehicles.get(self.vehicle_id)
        intersection = self.global_knowledge.world.get_intersection(intersection_id)

        if vehicle is None:
            LOGGER.warning("Ignoring position update for unknown taxi %s.", self.vehicle_id)
            return
        if intersection is None:
            LOGGER.warning(
                "Ignoring position update to unknown intersection %s for taxi %s.",
                intersection_id,
                self.vehicle_id,
            )
            return

        vehicle.position = intersection
        self.local_knowledge.record_monitor("record_vehicle_position", event)

    def reset(self) -> None:
        self.local_knowledge.reset()


class Analyze:


    def __init__(self, global_knowledge: GlobalKnowledge) -> None:
        self.global_knowledge = global_knowledge

    def build_problem(self) -> AssignmentProblemSnapshot:
        vehicles = self.global_knowledge.free_vehicles()
        current_time = self.global_knowledge.current_time
        requests = [
            request
            for request in self.global_knowledge.open_requests.values()
            if request.remaining_customers > 0
            and not self.global_knowledge.request_is_cancelled(request.id)
            and (
                (latest := self.global_knowledge.request_latest_service_time(request.id)) is None
                or current_time <= latest
            )
        ]
        choices_by_vehicle: list[list[AssignmentCandidate]] = [[] for _ in vehicles]
        settings = self.global_knowledge.settings
        allowed_by_request = {
            request.id: self._nearest_vehicle_indices(
                vehicles,
                request,
                settings.normalized().max_candidate_vehicles_per_request,
            )
            for request in requests
        }

        for vehicle_index, vehicle in enumerate(vehicles):
            if vehicle.position is None:
                continue
            if vehicle.max_capacity < 1:
                continue

            for request in requests:
                if vehicle_index not in allowed_by_request[request.id]:
                    continue
                pickup_candidates = self.global_knowledge.world.route_candidates(
                    vehicle.position.id,
                    request.start_intersection.id,
                    vehicle.properties,
                    settings.route_candidates_per_leg,
                    settings.route_search_limit,
                )
                destination_candidates = self.global_knowledge.world.route_candidates(
                    request.start_intersection.id,
                    request.end_intersection.id,
                    vehicle.properties,
                    settings.route_candidates_per_leg,
                    settings.route_search_limit,
                )
                if not pickup_candidates or not destination_candidates:
                    continue

                for to_pickup in pickup_candidates:
                    for to_destination in destination_candidates:
                        choices_by_vehicle[vehicle_index].append(
                            AssignmentCandidate(
                                vehicle=vehicle,
                                request=request,
                                to_pickup=to_pickup,
                                to_destination=to_destination,
                                travel_time=to_pickup.travel_time + to_destination.travel_time,
                                distance=to_pickup.distance + to_destination.distance,
                                co2_emissions=(
                                    to_pickup.co2_emissions + to_destination.co2_emissions
                                ),
                                distance_cost=to_pickup.distance_cost + to_destination.distance_cost,
                            )
                        )

        return AssignmentProblemSnapshot(
            vehicles=vehicles,
            requests=requests,
            choices_by_vehicle=choices_by_vehicle,
            current_time=self.global_knowledge.current_time,
        )

    def _nearest_vehicle_indices(
        self,
        vehicles: list[Vehicle],
        request: RideRequest,
        limit: int,
    ) -> set[int]:


        all_indices = set(range(len(vehicles)))
        if limit <= 0 or len(vehicles) <= limit:
            return all_indices

        world = self.global_knowledge.world
        scored: list[tuple[float, int]] = []
        for index, vehicle in enumerate(vehicles):
            if vehicle.position is None:
                continue
            distance = world.geo_distance(vehicle.position.id, request.start_intersection.id)
            scored.append((distance if distance is not None else float("inf"), index))

        if not scored or all(distance == float("inf") for distance, _ in scored):
            return all_indices

        scored.sort(key=lambda item: item[0])
        return {index for _, index in scored[:limit]}


class Reposition:


    def __init__(self, global_knowledge: GlobalKnowledge) -> None:
        self.global_knowledge = global_knowledge

    def idle_vehicles(self) -> list[Vehicle]:
        return [
            vehicle
            for vehicle in self.global_knowledge.free_vehicles()
            if vehicle.position is not None
        ]

    def _other_idle_positions(self, vehicle: Vehicle) -> list[int]:
        return [
            other.position.id
            for other in self.idle_vehicles()
            if other.id != vehicle.id and other.position is not None
        ]

    def _nearest_neighbor_distance(self, intersection_id: int, others: list[int]) -> float | None:
        world = self.global_knowledge.world
        distances = [
            distance
            for other_id in others
            if (distance := world.geo_distance(intersection_id, other_id)) is not None
        ]
        return min(distances) if distances else None

    def is_clustered(self, vehicle: Vehicle, other_positions: list[int] | None = None) -> bool:
        settings = self.global_knowledge.settings.normalized()
        if vehicle.position is None:
            return False
        others = other_positions if other_positions is not None else self._other_idle_positions(vehicle)
        nearest = self._nearest_neighbor_distance(vehicle.position.id, others)
        if nearest is None:
            return False
        return nearest < settings.idle_min_spacing_meters

    def reposition_target_id(
        self,
        vehicle: Vehicle,
        other_positions: list[int] | None = None,
    ) -> int | None:

        settings = self.global_knowledge.settings.normalized()
        world = self.global_knowledge.world
        if vehicle.position is None:
            return None

        others = other_positions if other_positions is not None else self._other_idle_positions(vehicle)
        if not others:
            return None

        current = self._nearest_neighbor_distance(vehicle.position.id, others)
        if current is None:
            return None

                                                                                               
        travel_budget = max(60.0, settings.idle_min_spacing_meters / 10.0)
        travel_times = world.travel_times_from(vehicle.position.id, vehicle.properties)

        best_id: int | None = None
        best_distance = current
        for intersection_id, travel_time in travel_times.items():
            if intersection_id == vehicle.position.id:
                continue
            if travel_time > travel_budget:
                continue
            candidate = self._nearest_neighbor_distance(intersection_id, others)
            if candidate is None:
                continue
            if candidate > best_distance:
                best_distance = candidate
                best_id = intersection_id

        return best_id


class ChromosomeEvaluator:


    def evaluate(
        self,
        chromosome: np.ndarray,
        snapshot: AssignmentProblemSnapshot,
    ) -> EvaluationResult:
        remaining_by_request = {
            request_index: request.remaining_customers
            for request_index, request in enumerate(snapshot.requests)
        }
        request_index_by_id = {request.id: index for index, request in enumerate(snapshot.requests)}
        decisions: list[AssignmentDecision] = []
        total_travel_time = 0.0
        total_distance = 0.0
        total_co2_emissions = 0.0
        total_distance_cost = 0.0
        total_late_service_time = 0.0
        duplicate_assignments = 0
        served_customers = 0
        deadline_violations = 0

        for vehicle_index, raw_gene in enumerate(chromosome.astype(int).tolist()):
            if vehicle_index >= len(snapshot.vehicles):
                continue

            choices = snapshot.choices_by_vehicle[vehicle_index]
            choice_index = raw_gene - 1
            if choice_index < 0 or choice_index >= len(choices):
                continue

            candidate = choices[choice_index]
            request_index = request_index_by_id[candidate.request.id]

            remaining = remaining_by_request.get(request_index, 0)
            if remaining <= 0:
                duplicate_assignments += 1
                continue

            served = min(candidate.vehicle.max_capacity, remaining)
            if served <= 0:
                continue

            remaining_by_request[request_index] = remaining - served
            served_customers += served
            pickup_time = snapshot.current_time + candidate.to_pickup.travel_time
            service_start_time = max(pickup_time, candidate.request.earliest_service_time)
            dropoff_time = service_start_time + candidate.to_destination.travel_time
            requested_pickup_time = max(snapshot.current_time, candidate.request.earliest_service_time)

            total_travel_time += max(0.0, dropoff_time - requested_pickup_time)
            total_distance += candidate.distance
            total_co2_emissions += candidate.co2_emissions
            total_distance_cost += candidate.distance_cost
            total_late_service_time += max(0.0, dropoff_time - candidate.request.latest_service_time)
            if dropoff_time > candidate.request.latest_service_time:
                deadline_violations += served
            decisions.append(
                AssignmentDecision(
                    vehicle=candidate.vehicle,
                    request=candidate.request,
                    to_pickup=candidate.to_pickup,
                    to_destination=candidate.to_destination,
                    served_customers=served,
                    travel_time=candidate.travel_time,
                    distance=candidate.distance,
                    co2_emissions=candidate.co2_emissions,
                    distance_cost=candidate.distance_cost,
                )
            )

        return EvaluationResult(
            decisions=decisions,
            unserved_customers=sum(remaining_by_request.values()) + duplicate_assignments,
            total_travel_time=total_travel_time,
            total_distance=total_distance,
            total_co2_emissions=total_co2_emissions,
            total_distance_cost=total_distance_cost,
            total_late_service_time=total_late_service_time,
            duplicate_assignments=duplicate_assignments,
            served_customers=served_customers,
            deadline_violations=deadline_violations,
        )


class _AssignmentProblem(Problem):
    def __init__(
        self,
        snapshot: AssignmentProblemSnapshot,
        evaluators: list[ChromosomeEvaluator],
        initial_chromosomes: list[np.ndarray],
    ) -> None:
        super().__init__(
            n_var=snapshot.chromosome_length,
            n_obj=4,
            n_ieq_constr=0,
            xl=np.zeros(snapshot.chromosome_length, dtype=int),
            xu=snapshot.upper_bounds,
            vtype=int,
        )
        self.snapshot = snapshot
        self.evaluators = evaluators
        self.initial_chromosomes = initial_chromosomes
        self.evaluation_cache: dict[tuple[int, ...], EvaluationResult] = {}

    def _evaluate(self, x_values: np.ndarray, out: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        objectives: list[tuple[float, float, float, float]] = []
        for index, chromosome in enumerate(x_values):
            key = _chromosome_key(chromosome)
            if key not in self.evaluation_cache:
                evaluator = self.evaluators[index % len(self.evaluators)]
                self.evaluation_cache[key] = evaluator.evaluate(chromosome, self.snapshot)
            objectives.append(self.evaluation_cache[key].objectives)

        out["F"] = np.array(objectives, dtype=float)


class _AssignmentElementwiseProblem(ElementwiseProblem):
    def __init__(
        self,
        snapshot: AssignmentProblemSnapshot,
        evaluator: ChromosomeEvaluator,
        initial_chromosomes: list[np.ndarray],
    ) -> None:
        super().__init__(
            n_var=snapshot.chromosome_length,
            n_obj=4,
            n_ieq_constr=0,
            xl=np.zeros(snapshot.chromosome_length, dtype=int),
            xu=snapshot.upper_bounds,
            vtype=int,
        )
        self.snapshot = snapshot
        self.evaluator = evaluator
        self.initial_chromosomes = initial_chromosomes
        self.evaluation_cache: dict[tuple[int, ...], EvaluationResult] = {}

    def _evaluate(self, x: np.ndarray, out: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        key = _chromosome_key(x)
        if key not in self.evaluation_cache:
            self.evaluation_cache[key] = self.evaluator.evaluate(x, self.snapshot)
        out["F"] = np.array(self.evaluation_cache[key].objectives, dtype=float)


class _WarmStartSampling(Sampling):
    def _do(self, problem: Problem, n_samples: int, **kwargs: Any) -> np.ndarray:
        if not hasattr(problem, "snapshot") or not hasattr(problem, "initial_chromosomes"):
            return np.zeros((n_samples, problem.n_var), dtype=int)

        rows = [problem.snapshot.repair_chromosome(chromosome) for chromosome in problem.initial_chromosomes]
        random_state = kwargs.get("random_state") or np.random.default_rng()

        while len(rows) < n_samples:
            rows.append(self._random_chromosome(problem.snapshot, random_state))

        return np.array(rows[:n_samples], dtype=int)

    @staticmethod
    def _random_chromosome(snapshot: AssignmentProblemSnapshot, random_state: Any) -> np.ndarray:
        genes: list[int] = []
        for upper_bound in snapshot.upper_bounds.tolist():
            if hasattr(random_state, "integers"):
                genes.append(int(random_state.integers(0, upper_bound + 1)))
            else:
                genes.append(int(random_state.randint(0, upper_bound + 1)))
        return np.array(genes, dtype=int)


class Plan:


    def __init__(self, global_knowledge: GlobalKnowledge, evaluators: list[ChromosomeEvaluator]) -> None:
        self.global_knowledge = global_knowledge
        self.evaluators = evaluators

    def plan(self, snapshot: AssignmentProblemSnapshot) -> EvaluationResult:

        if snapshot.empty:
            return EvaluationResult(
                [],
                sum(request.remaining_customers for request in snapshot.requests),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0,
                0,
            )

        settings = self.global_knowledge.settings.normalized()
        initial_chromosomes = self._initial_chromosomes(snapshot)
        evaluator = self.evaluators[0]
        evaluation_cache: dict[tuple[int, ...], EvaluationResult] = {}
        chromosomes: list[np.ndarray]

        search_space_size = _search_space_size(snapshot)
        if (
            settings.trivial_search_threshold > 0
            and search_space_size <= settings.trivial_search_threshold
        ):
            chromosomes = self._unique_chromosomes(
                _enumerate_chromosomes(snapshot) + initial_chromosomes,
                snapshot.chromosome_length,
            )
            for chromosome in chromosomes:
                key = _chromosome_key(chromosome)
                if key not in evaluation_cache:
                    evaluation_cache[key] = evaluator.evaluate(chromosome, snapshot)
        else:
            algorithm = NSGA2(
                pop_size=settings.population_size,
                sampling=_WarmStartSampling(),
                crossover=SBX(prob=0.9, eta=15, repair=RoundingRepair()),
                mutation=PM(eta=20, repair=RoundingRepair()),
                eliminate_duplicates=True,
            )
            termination = ("n_gen", settings.generations)

            if settings.eval_workers > 1:
                elementwise_problem = _AssignmentElementwiseProblem(
                    snapshot,
                    evaluator,
                    initial_chromosomes,
                )
                with Pool(processes=settings.eval_workers) as pool:
                    runner = StarmapParallelization(pool.starmap)
                    result = minimize(
                        elementwise_problem,
                        algorithm,
                        termination,
                        seed=settings.seed,
                        verbose=False,
                        elementwise_runner=runner,
                    )
                evaluation_cache = elementwise_problem.evaluation_cache
            else:
                problem = _AssignmentProblem(
                    snapshot,
                    self.evaluators,
                    initial_chromosomes,
                )
                result = minimize(
                    problem,
                    algorithm,
                    termination,
                    seed=settings.seed,
                    verbose=False,
                )
                evaluation_cache = problem.evaluation_cache

            chromosomes = self._result_chromosomes(result.X)
            if result.pop is not None:
                chromosomes.extend(self._result_chromosomes(result.pop.get("X")))
            chromosomes.extend(initial_chromosomes)
            chromosomes = self._unique_chromosomes(chromosomes, snapshot.chromosome_length)

        evaluations = [
            self._cached_evaluation(evaluation_cache, chromosome, snapshot, index)
            for index, chromosome in enumerate(chromosomes)
        ]
        best = min(evaluations, key=self._utility, default=None)
        if best is None:
            return EvaluationResult(
                [],
                sum(request.remaining_customers for request in snapshot.requests),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0,
                0,
            )

        self._cache_chromosomes(chromosomes, best, snapshot)
        self.global_knowledge.last_objectives = best.objectives
        return best

    def _cached_evaluation(
        self,
        evaluation_cache: dict[tuple[int, ...], EvaluationResult],
        chromosome: np.ndarray,
        snapshot: AssignmentProblemSnapshot,
        index: int,
    ) -> EvaluationResult:
        key = _chromosome_key(chromosome)
        cached = evaluation_cache.get(key)
        if cached is not None:
            return cached
        return self.evaluators[index % len(self.evaluators)].evaluate(chromosome, snapshot)

    def choose(self, snapshot: AssignmentProblemSnapshot) -> EvaluationResult:
        return self.plan(snapshot)

    def _utility(self, result: EvaluationResult) -> float:
        settings = self.global_knowledge.settings.normalized()
        return (
            result.unserved_customers * settings.unserved_customer_penalty
            + result.total_travel_time * settings.time_weight
            + result.total_late_service_time * settings.late_service_penalty
            + result.deadline_violations * settings.late_service_penalty
            + result.total_distance_cost * settings.cost_weight
            + result.total_co2_emissions * settings.co2_weight
            - result.served_customers * settings.throughput_weight
        )

    @staticmethod
    def _result_chromosomes(raw_result: Any) -> list[np.ndarray]:
        array = np.asarray(raw_result, dtype=int)
        if array.size == 0:
            return []
        if array.ndim == 1:
            return [array]
        return [row for row in array]

    @staticmethod
    def _seed_chromosomes(snapshot: AssignmentProblemSnapshot) -> list[np.ndarray]:
        chromosomes: list[np.ndarray] = []
        for vehicle_index, choices in enumerate(snapshot.choices_by_vehicle):
            for choice_index in range(len(choices)):
                chromosome = np.zeros(snapshot.chromosome_length, dtype=int)
                chromosome[vehicle_index] = choice_index + 1
                chromosomes.append(chromosome)
        return chromosomes

    def _initial_chromosomes(self, snapshot: AssignmentProblemSnapshot) -> list[np.ndarray]:
        chromosomes: list[np.ndarray] = []
        if self.global_knowledge.best_chromosome is not None:
            chromosomes.append(snapshot.repair_cached_chromosome(self.global_knowledge.best_chromosome))
        chromosomes.extend(
            snapshot.repair_cached_chromosome(cached)
            for cached in self.global_knowledge.cached_chromosomes
        )
        chromosomes.extend(self._seed_chromosomes(snapshot))
        return self._unique_chromosomes(chromosomes, snapshot.chromosome_length)

    def _cache_chromosomes(
        self,
        chromosomes: list[np.ndarray],
        best: EvaluationResult,
        snapshot: AssignmentProblemSnapshot,
    ) -> None:
        settings = self.global_knowledge.settings.normalized()
        unique = self._unique_chromosomes(chromosomes, snapshot.chromosome_length)
        self.global_knowledge.cached_chromosomes = [
            CachedChromosome.from_snapshot(chromosome, snapshot)
            for chromosome in unique[: settings.warm_start_cache_size]
        ]

        best_chromosome = self._chromosome_for_result(best, snapshot)
        self.global_knowledge.best_chromosome = (
            CachedChromosome.from_snapshot(best_chromosome, snapshot)
            if best_chromosome is not None
            else None
        )

    @staticmethod
    def _unique_chromosomes(chromosomes: list[np.ndarray], chromosome_length: int) -> list[np.ndarray]:
        unique: list[np.ndarray] = []
        seen: set[tuple[int, ...]] = set()
        for chromosome in chromosomes:
            repaired = np.asarray(chromosome, dtype=int).flatten()
            if len(repaired) != chromosome_length:
                continue
            key = tuple(int(gene) for gene in repaired.tolist())
            if key in seen:
                continue
            seen.add(key)
            unique.append(repaired.copy())
        return unique

    @staticmethod
    def _chromosome_for_result(
        result: EvaluationResult,
        snapshot: AssignmentProblemSnapshot,
    ) -> np.ndarray | None:
        chromosome = np.zeros(snapshot.chromosome_length, dtype=int)
        vehicle_index_by_id = {vehicle.id: index for index, vehicle in enumerate(snapshot.vehicles)}
        choice_keys = snapshot.choice_keys_by_vehicle

        for decision in result.decisions:
            vehicle_index = vehicle_index_by_id.get(decision.vehicle.id)
            if vehicle_index is None:
                continue
            decision_key = AssignmentChoiceKey(
                request_id=decision.request.id,
                pickup_road_ids=decision.to_pickup.road_ids,
                destination_road_ids=decision.to_destination.road_ids,
            )
            try:
                chromosome[vehicle_index] = choice_keys[vehicle_index].index(decision_key) + 1
            except ValueError:
                continue

        if not np.any(chromosome):
            return None
        return chromosome


class TaxiExecute:


    def __init__(
        self,
        vehicle_id: str,
        global_knowledge: GlobalKnowledge,
        local_knowledge: LocalKnowledge,
    ) -> None:
        self.vehicle_id = vehicle_id
        self.global_knowledge = global_knowledge
        self.local_knowledge = local_knowledge

    def apply_decision(self, decision: AssignmentDecision) -> list[Event]:
        if decision.vehicle.id != self.vehicle_id:
            return []

        request = decision.request
        if self.global_knowledge.request_is_cancelled(request.id):
            return []
        latest_service_time = self.global_knowledge.request_latest_service_time(request.id)
        if latest_service_time is not None and self.global_knowledge.current_time > latest_service_time:
            return []

        if decision.vehicle.position is None:
            return []

        route_steps = self._route_steps(decision)
        world = self.global_knowledge.world
        if world.roads and not world.route_steps_have_outgoing_at_all_nodes(
            decision.vehicle.position.id,
            route_steps,
        ):
            LOGGER.warning(
                "Skipping route for %s on %s; not all intersections have outgoing roads.",
                self.vehicle_id,
                decision.request.id,
            )
            return []

        response = Event("taxi-fleet", "plan-route")
        response.put("move-id", self.global_knowledge.next_move_id())
        response.put("route", route_steps)
        response.put("vehicle-id", decision.vehicle.id)
        response.put("request-id", decision.request.id)
        response.explanations = self._explanations(decision)
        response.put("explanations", response.explanations)

        self.global_knowledge.mark_request_assigned(decision.request.id)
        self.global_knowledge.bind_request_vehicle(decision.request.id, decision.vehicle.id)
        decision.vehicle.is_busy = True
        decision.vehicle.last_active_time = self.global_knowledge.current_time
        decision.request.remaining_customers -= decision.served_customers
        self.global_knowledge.remember_request_deadline(
            decision.request.id,
            decision.request.latest_service_time,
        )
        if decision.request.remaining_customers <= 0:
            self.global_knowledge.open_requests.pop(decision.request.id, None)

        self.local_knowledge.record_execution(decision)
        LOGGER.info("Taxi %s executing assignment for %s.", self.vehicle_id, decision.request.id)
        return [response]

    def _route_steps(self, decision: AssignmentDecision) -> list[dict[str, Any]]:
        steps = self._to_simulation_steps(decision.to_pickup)
        steps.append(
            {
                "type": "pick-up-passengers",
                "request-id": decision.request.id,
                "intersection-id": decision.request.start_intersection.id,
                "count": decision.served_customers,
            }
        )
        steps.extend(self._to_simulation_steps(decision.to_destination))
        steps.append(
            {
                "type": "drop-off-passengers",
                "request-id": decision.request.id,
                "intersection-id": decision.request.end_intersection.id,
                "count": decision.served_customers,
            }
        )
        return steps

    @staticmethod
    def _to_simulation_steps(candidate: RouteCandidate) -> list[dict[str, Any]]:
        return [road.to_follow_road_step() for road in candidate.path.edge_list]

    def _explanations(self, decision: AssignmentDecision) -> dict[str, str]:
        settings = self.global_knowledge.settings
        motor_type = decision.vehicle.properties.get("motor-type", "electric")
        is_electric = motor_type == "electric"
        vehicle_desc = "eco-friendly electric taxi" if is_electric else "standard combustion-engine taxi"

        pickup_sec = int(decision.to_pickup.travel_time)
        trip_sec = int(decision.to_destination.travel_time)
        co2 = decision.co2_emissions

        pickup_desc = f"{pickup_sec} seconds" if pickup_sec < 60 else f"{pickup_sec // 60}m {pickup_sec % 60}s"
        trip_desc = f"{trip_sec} seconds" if trip_sec < 60 else f"{trip_sec // 60}m {trip_sec % 60}s"

        if is_electric:
            customer_explanation = (
                f"Your ride with {decision.vehicle.id} has been booked! We selected this {vehicle_desc} "
                f"for you. It will arrive in about {pickup_desc}, and your trip will take {trip_desc}. "
                f"This choice has a very low carbon footprint of {co2:.1f} CO2 units!"
            )
        else:
            customer_explanation = (
                f"Your ride with {decision.vehicle.id} has been booked! This {vehicle_desc} "
                f"will pick you up in about {pickup_desc}, and your trip will take {trip_desc}, "
                f"carefully balancing travel speed and emission efficiency ({co2:.1f} CO2 units)."
            )

        pickup_id = decision.request.start_intersection.id
        dropoff_id = decision.request.end_intersection.id
        passengers = decision.served_customers
        pickup_dist = decision.to_pickup.distance
        dest_dist = decision.to_destination.distance

        taxi_driver_explanation = (
            f"Dispatch: Proceed to intersection {pickup_id} to pick up {passengers} passenger(s) "
            f"(distance: {pickup_dist:.0f}m, ETA: {pickup_sec}s). "
            f"Then, navigate to intersection {dropoff_id} to drop them off "
            f"(distance: {dest_dist:.0f}m, travel time: {trip_sec}s)."
        )

        ice_status = "electric (Non-ICE)" if is_electric else "combustion (ICE)"
        taxi_manager_explanation = (
            f"Selected {decision.vehicle.id} ({ice_status}) for request {decision.request.id} ({passengers} passengers). "
            f"Goal weights: Time={settings.time_weight:.2f}, CO2={settings.co2_weight:.2f}, Throughput={settings.throughput_weight:.2f}. "
            f"Metrics: arrival time {decision.travel_time:.0f}s, distance {decision.distance:.0f}m, emissions {co2:.1f} CO2 units, cost {decision.distance_cost:.2f}."
        )

        return {
            "customer": customer_explanation,
            "taxi-manager": taxi_manager_explanation,
            "taxi-driver": taxi_driver_explanation,
        }


class TaxiSlave:


    def __init__(self, vehicle_id: str, global_knowledge: GlobalKnowledge) -> None:
        self.vehicle_id = vehicle_id
        self.global_knowledge = global_knowledge
        self.local = LocalKnowledge(vehicle_id=vehicle_id)
        self.monitor = TaxiMonitor(vehicle_id, global_knowledge, self.local)
        self.execute = TaxiExecute(vehicle_id, global_knowledge, self.local)


                            
EvolutionarySlave = TaxiSlave
MapeKSlave = TaxiSlave


class MasterEvolutionaryLoop:


    def __init__(
        self,
        world: WorldManager,
        settings: EvolutionarySettings | None = None,
        *,
        distributed: bool = False,
    ) -> None:
        normalized_settings = (settings or EvolutionarySettings()).normalized()
        self.distributed = distributed
        self.global_knowledge = GlobalKnowledge(world=world, settings=normalized_settings)
        self.dispatch_monitor = DispatchMonitor(self.global_knowledge)
        self.analyze = Analyze(self.global_knowledge)
        evaluators = [ChromosomeEvaluator() for _ in range(normalized_settings.slave_count)]
        self._planner = Plan(self.global_knowledge, evaluators)
        self._reposition = Reposition(self.global_knowledge)
        self.taxi_slaves: dict[str, TaxiSlave] = {}
        self._taxi_pool = None
        self._last_adapt_time: int | None = None
        self._reposition_targets: dict[str, int] = {}

        if distributed:
            from .slave_pool import TaxiProcessPool

            self._taxi_pool = TaxiProcessPool()

    @property
    def slaves(self) -> list[TaxiSlave]:
        return list(self.taxi_slaves.values())

    @property
    def knowledge(self) -> GlobalKnowledge:

        return self.global_knowledge

    def local_knowledge_for(self, vehicle_id: str) -> LocalKnowledge | None:
        if self._taxi_pool is not None:
            return self._taxi_pool.local_knowledge_for(vehicle_id)
        taxi = self.taxi_slaves.get(vehicle_id)
        return taxi.local if taxi is not None else None

    @property
    def slave_count(self) -> int:
        return len(self.taxi_slaves)

    def close(self) -> None:
        if self._taxi_pool is not None:
            self._taxi_pool.shutdown()
            self._taxi_pool = None

    def __enter__(self) -> "MasterEvolutionaryLoop":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def vehicles(self) -> dict[str, Vehicle]:
        return self.global_knowledge.vehicles

    @property
    def open_requests(self) -> dict[str, RideRequest]:
        return self.global_knowledge.open_requests

    def plan_route(self, event: Event) -> list[Event]:
        self.dispatch_monitor.record_request(event)
        return self._adapt(force=True)

    def add_vehicle(self, event: Event) -> list[Event]:
        vehicle_id = str(event.get("id"))
        self._register_taxi(vehicle_id)
        self._taxi_monitor(vehicle_id, "record_vehicle", event)
        responses = self._adapt(force=False)
        responses.extend(self._maybe_reposition_after(responses, vehicle_id))
        return responses

    def vehicle_available(self, event: Event) -> list[Event]:
        vehicle_id = str(event.get("vehicle-id"))
        self._taxi_monitor(vehicle_id, "record_vehicle_available", event)
        responses = self._adapt(force=False)
        responses.extend(self._maybe_reposition_after(responses, vehicle_id))
        return responses

    def update_vehicle_position(self, event: Event) -> list[Event]:
        vehicle_id = str(event.get("vehicle-id"))
        self._taxi_monitor(vehicle_id, "record_vehicle_position", event)
        vehicle = self.global_knowledge.vehicles.get(vehicle_id)
        target_id = self._reposition_targets.get(vehicle_id)
        if vehicle is not None and vehicle.position is not None and vehicle.position.id == target_id:
            self._reposition_targets.pop(vehicle_id, None)
        return []

    def update_time(self, event: Event) -> list[Event]:
        self.dispatch_monitor.record_time(event)
        responses = self._expire_overdue_requests()
        if self._should_reoptimize_on_time():
            responses.extend(self._adapt(force=False))
        return responses

    def road_network_changed(self, event: Event) -> list[Event]:
        if not self.global_knowledge.open_requests or not self.global_knowledge.free_vehicles():
            return []
        return self._adapt(force=False)

    def on_reset(self, event: Event) -> list[Event]:
        self.dispatch_monitor.reset()
        for taxi in self.taxi_slaves.values():
            taxi.monitor.reset()
        if self._taxi_pool is not None:
            self._taxi_pool.reset_local_knowledge()
        self.taxi_slaves.clear()
        self._last_adapt_time = None
        self._reposition_targets.clear()
        return []

    def remove_request(self, event: Event) -> list[Event]:
        self.dispatch_monitor.remove_request(event)
        return []

    def record_person(self, event: Event) -> list[Event]:
        self.dispatch_monitor.record_person(event)
        return []

    def record_pickup(self, event: Event) -> list[Event]:
        request_id = event.get("request-id")
        if request_id is not None:
            self.global_knowledge.mark_request_picked_up(str(request_id))
        return []

    def cancel_request(self, event: Event) -> list[Event]:
        if event.category == "person" and event.name == "removed":
            return []
        request_id = self._request_id_from_cancel_event(event)
        if request_id is None:
            return []
        return self._cancel_request_dispatch(request_id, f"{event.category}:{event.name}")

    def _request_id_from_cancel_event(self, event: Event) -> str | None:
        request_id = event.get("request-id")
        if request_id is not None:
            return str(request_id)

        if event.category == "person" and event.name == "removed":
            person_id = event.get("id")
            if person_id is None:
                return None
            return self.global_knowledge.resolve_request_for_person(str(person_id))

        if event.category == "request" and event.get("id") is not None:
            return str(event.get("id"))

        return None

    def _cancel_request_dispatch(self, request_id: str, source: str) -> list[Event]:
        if self.global_knowledge.request_is_picked_up(request_id):
            return []
        vehicle_id = self.global_knowledge.request_active_vehicle.get(request_id)
        self.dispatch_monitor.cancel_request(request_id)
        responses: list[Event] = []
        if vehicle_id is not None:
            vehicle = self.global_knowledge.vehicles.get(vehicle_id)
            if vehicle is not None and vehicle.is_busy:
                                                                                          
                                                                             
                responses.append(self._hold_event(vehicle_id))
                vehicle.is_busy = False

        return responses

    def _expire_overdue_requests(self) -> list[Event]:
        current_time = self.global_knowledge.current_time
        responses: list[Event] = []
        tracked_request_ids = set(self.global_knowledge.open_requests) | set(
            self.global_knowledge.request_active_vehicle
        )
        for request_id in tracked_request_ids:
            if self.global_knowledge.request_is_cancelled(request_id):
                continue
            if self.global_knowledge.request_is_picked_up(request_id):
                continue
            latest_service_time = self.global_knowledge.request_latest_service_time(request_id)
            if latest_service_time is not None and current_time > latest_service_time:
                responses.extend(self._cancel_request_dispatch(request_id, "time:expired"))
        return responses

    def _register_taxi(self, vehicle_id: str) -> None:
        if vehicle_id in self.taxi_slaves or (
            self._taxi_pool is not None and self._taxi_pool.has_taxi(vehicle_id)
        ):
            return

        if self._taxi_pool is not None:
            self._taxi_pool.register(vehicle_id)
            return

        self.taxi_slaves[vehicle_id] = TaxiSlave(vehicle_id, self.global_knowledge)

    def _taxi_monitor(self, vehicle_id: str, operation: str, event: Event) -> None:
        if self._taxi_pool is not None:
            self._register_taxi(vehicle_id)
            self._taxi_pool.monitor(
                vehicle_id,
                self.global_knowledge,
                self.global_knowledge.world,
                operation,                          
                event,
            )
            return

        taxi = self.taxi_slaves.get(vehicle_id)
        if taxi is None:
            LOGGER.warning("No taxi slave registered for %s.", vehicle_id)
            return

        if operation == "reset":
            taxi.monitor.reset()
            return
        getattr(taxi.monitor, operation)(event)

    def plan(
        self,
        vehicles: list[Vehicle] | None = None,
        requests: list[RideRequest] | None = None,
        choices_by_vehicle: list[list[AssignmentCandidate]] | None = None,
    ) -> EvaluationResult:

        if vehicles is None or requests is None or choices_by_vehicle is None:
            snapshot = self.analyze.build_problem()
        else:
            snapshot = AssignmentProblemSnapshot(
                vehicles=vehicles,
                requests=requests,
                choices_by_vehicle=choices_by_vehicle,
                current_time=self.global_knowledge.current_time,
            )
        return self._planner.plan(snapshot)

    def _decision_route_steps(self, decision: AssignmentDecision) -> list[dict[str, Any]]:
        return TaxiExecute(
            decision.vehicle.id,
            self.global_knowledge,
            LocalKnowledge(vehicle_id=decision.vehicle.id),
        )._route_steps(decision)

    def _decision_route_is_valid(self, decision: AssignmentDecision) -> bool:
        vehicle = decision.vehicle
        if vehicle.position is None:
            return False
        return self.global_knowledge.world.route_steps_have_outgoing_at_all_nodes(
            vehicle.position.id,
            self._decision_route_steps(decision),
        )

    def _execute_decisions(self, decisions: list[AssignmentDecision]) -> list[Event]:
        responses: list[Event] = []
        skipped = {"cancelled": 0, "expired": 0, "no_customers": 0, "invalid_route": 0, "no_slave": 0}
        for decision in decisions:
                                                                                        
                                                                                         
            request = decision.request
            if self.global_knowledge.request_is_cancelled(request.id):
                skipped["cancelled"] += 1
                continue
            latest_service_time = self.global_knowledge.request_latest_service_time(request.id)
            if latest_service_time is not None and self.global_knowledge.current_time > latest_service_time:
                skipped["expired"] += 1
                continue
            if request.remaining_customers <= 0:
                skipped["no_customers"] += 1
                continue
            if not self._decision_route_is_valid(decision):
                skipped["invalid_route"] += 1
                LOGGER.warning(
                    "Skipping route for %s on %s; not all intersections have outgoing roads.",
                    decision.vehicle.id,
                    request.id,
                )
                continue

            vehicle_id = decision.vehicle.id
            self._reposition_targets.pop(vehicle_id, None)
            if self._taxi_pool is not None:
                self._register_taxi(vehicle_id)
                responses.extend(
                    self._taxi_pool.execute_decision(
                        vehicle_id,
                        self.global_knowledge,
                        self.global_knowledge.world,
                        decision,
                    )
                )
                continue

            taxi = self.taxi_slaves.get(vehicle_id)
            if taxi is None:
                skipped["no_slave"] += 1
                LOGGER.warning("Cannot execute plan; no taxi slave for %s.", vehicle_id)
                continue
            responses.extend(taxi.execute.apply_decision(decision))
        if skipped and any(v > 0 for v in skipped.values()):
            LOGGER.debug("Execute: %d applied, skipped=%s", len(responses), skipped)
        return responses

    def _plan_route_event(self, vehicle_id: str, route_steps: list[dict[str, Any]]) -> Event | None:
        if route_steps:
            vehicle = self.global_knowledge.vehicles.get(vehicle_id)
            if vehicle is None or vehicle.position is None:
                LOGGER.warning(
                    "Skipping route for %s; vehicle position is unknown.",
                    vehicle_id,
                )
                return None
            if not self.global_knowledge.world.route_steps_have_outgoing_at_all_nodes(
                vehicle.position.id,
                route_steps,
            ):
                LOGGER.warning(
                    "Skipping route for %s; not all intersections have outgoing roads.",
                    vehicle_id,
                )
                return None

        response = Event("taxi-fleet", "plan-route")
        response.put("move-id", self.global_knowledge.next_move_id())
        response.put("route", route_steps)
        response.put("vehicle-id", vehicle_id)
        if route_steps:
            vehicle = self.global_knowledge.vehicles.get(vehicle_id)
            if vehicle is not None:
                vehicle.is_busy = True
        return response

    def _hold_event(self, vehicle_id: str) -> Event:

        event = self._plan_route_event(vehicle_id, [])
        if event is None:
            response = Event("taxi-fleet", "plan-route")
            response.put("move-id", self.global_knowledge.next_move_id())
            response.put("route", [])
            response.put("vehicle-id", vehicle_id)
            return response
        return event

    def _maybe_reposition_after(self, responses: list[Event], vehicle_id: str) -> list[Event]:

        settings = self.global_knowledge.settings.normalized()
        if not settings.enable_idle_repositioning or settings.max_repositions_per_event <= 0:
            return []

                                                                                 
        has_open_requests = any(
            request.remaining_customers > 0
            for request in self.global_knowledge.open_requests.values()
        )

        planned_responses: list[Event] = []
        excluded_vehicle_ids = {
            str(response.get("vehicle-id"))
            for response in responses
            if response.get("vehicle-id") is not None
        }
        candidate_ids = self._reposition_candidate_ids(vehicle_id, excluded_vehicle_ids)

        for candidate_id in candidate_ids:
            if len(planned_responses) >= settings.max_repositions_per_event:
                break

            vehicle = self.global_knowledge.vehicles.get(candidate_id)
            if vehicle is None or vehicle.position is None:
                continue

                                                                               
                                                
            if has_open_requests:
                idle_time = self.global_knowledge.current_time - vehicle.last_active_time
                if idle_time < settings.idle_reposition_threshold_seconds:
                    continue

            other_positions = self._effective_idle_positions(candidate_id)
            if not self._reposition.is_clustered(vehicle, other_positions):
                continue

            target_id = self._reposition.reposition_target_id(vehicle, other_positions)
            if target_id is None:
                continue

            route = self.global_knowledge.world.shortest_path_route(
                vehicle.position.id,
                target_id,
                vehicle.properties,
            )
            if route is None or not route.path.edge_list:
                continue

            steps = [road.to_follow_road_step() for road in route.path.edge_list]
            LOGGER.info("Repositioning idle taxi %s to intersection %s.", candidate_id, target_id)
            self._reposition_targets[candidate_id] = target_id
            route_event = self._plan_route_event(candidate_id, steps)
            if route_event is not None:
                planned_responses.append(route_event)

        return planned_responses

    def _reposition_candidate_ids(
        self,
        priority_vehicle_id: str,
        excluded_vehicle_ids: set[str],
    ) -> list[str]:
        vehicles = [
            vehicle
            for vehicle in self.global_knowledge.free_vehicles()
            if vehicle.position is not None
            and vehicle.id not in excluded_vehicle_ids
            and vehicle.id not in self._reposition_targets
        ]
        vehicles.sort(key=lambda vehicle: (vehicle.id != priority_vehicle_id, vehicle.id))
        return [vehicle.id for vehicle in vehicles]

    def _effective_idle_positions(self, excluded_vehicle_id: str) -> list[int]:
        positions: list[int] = []
        for vehicle in self.global_knowledge.free_vehicles():
            if vehicle.id == excluded_vehicle_id:
                continue
            if vehicle.id in self._reposition_targets:
                positions.append(self._reposition_targets[vehicle.id])
            elif vehicle.position is not None:
                positions.append(vehicle.position.id)
        return positions

    def _adapt(self, force: bool = False) -> list[Event]:
        settings = self.global_knowledge.settings.normalized()
        open_requests = {
            request_id: request.remaining_customers
            for request_id, request in self.global_knowledge.open_requests.items()
            if request.remaining_customers > 0
        }
        free_vehicle_count = len(self.global_knowledge.free_vehicles())
        LOGGER.debug(
            "Adapt: %d open requests, %d free vehicles, force=%s, t=%d",
            len(open_requests),
            free_vehicle_count,
            force,
            self.global_knowledge.current_time,
        )
        debounced = (
            not force
            and settings.replan_debounce_seconds > 0
            and self._last_adapt_time is not None
            and self.global_knowledge.current_time - self._last_adapt_time
            < settings.replan_debounce_seconds
        )
        pending_assignment = bool(open_requests) and free_vehicle_count > 0
        if debounced and pending_assignment:
            debounced = False
        if debounced:
            return []

        snapshot = self.analyze.build_problem()
        LOGGER.debug(
            "Analyze: %d vehicles, %d requests, candidates: %d",
            len(snapshot.vehicles),
            len(snapshot.requests),
            sum(len(c) for c in snapshot.choices_by_vehicle),
        )
        result = self.plan(snapshot.vehicles, snapshot.requests, snapshot.choices_by_vehicle)
        LOGGER.debug(
            "Plan: %d decisions, unserved=%d",
            len(result.decisions),
            result.unserved_customers,
        )
        if snapshot.vehicles and snapshot.requests:
            self.global_knowledge.last_reoptimization_time = self.global_knowledge.current_time
        self._last_adapt_time = self.global_knowledge.current_time
        responses = self._execute_decisions(result.decisions)

        LOGGER.info(
            "Dispatch executed %s assignments; %s requests remain open.",
            len(responses),
            len(self.global_knowledge.open_requests),
        )
        return responses

    def _should_reoptimize_on_time(self) -> bool:
        interval = self.global_knowledge.settings.normalized().continuous_reoptimization_interval
        if interval <= 0:
            return False
        if not self.global_knowledge.open_requests or not self.global_knowledge.free_vehicles():
            return False
        if self.global_knowledge.last_reoptimization_time is None:
            return True
        return self.global_knowledge.current_time - self.global_knowledge.last_reoptimization_time >= interval


MasterMapeKLoop = MasterEvolutionaryLoop
EvolutionaryOptimizer = MasterEvolutionaryLoop
EvolutionaryMapeKOptimizer = MasterEvolutionaryLoop
