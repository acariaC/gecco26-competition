import numpy as np

from src.config import EvolutionarySettings
from src.dispatcher import EventDispatcher
from src.events import Event
from src.models import AssignmentDecision
from src.optimizer import CachedChromosome, MasterEvolutionaryLoop, MasterMapeKLoop
from src.world import WorldManager

from tests.test_world import _roads_with_return_edges


def settings() -> EvolutionarySettings:
    return EvolutionarySettings(
        population_size=24,
        generations=12,
        seed=11,
        slave_count=2,
        replan_debounce_seconds=0,
    )


def test_contextual_goal_weights_default_to_pdf_values() -> None:
    settings = EvolutionarySettings()
    assert settings.time_weight == 0.4
    assert settings.co2_weight == 0.4
    assert settings.throughput_weight == 0.2
    assert settings.cost_weight == 0.0


def fast_settings(**overrides: int | float | bool) -> EvolutionarySettings:
    values = {
        "population_size": 4,
        "generations": 1,
        "seed": 11,
        "slave_count": 1,
        "continuous_reoptimization_interval": 5,
        "replan_debounce_seconds": 0,
    }
    values.update(overrides)
    return EvolutionarySettings(**values)


def build_world() -> WorldManager:
    world = WorldManager("http://localhost:8088")
    world.add_intersections([{"id": 1}, {"id": 2}, {"id": 3}])
    world.add_roads(
        _roads_with_return_edges(
            [
                {"id": 10, "start-node": 1, "end-node": 2, "length": 1000.0, "maximum-speed": 60},
                {"id": 20, "start-node": 2, "end-node": 3, "length": 1000.0, "maximum-speed": 60},
                {"id": 30, "start-node": 1, "end-node": 3, "length": 4000.0, "maximum-speed": 60},
            ]
        )
    )
    return world


def add_vehicle(
    optimizer: MasterEvolutionaryLoop,
    vehicle_id: str,
    intersection_id: int,
    capacity: int = 4,
    properties: dict[str, float | int | str] | None = None,
) -> list[Event]:
    vehicle_properties = {"maximum-capacity": capacity}
    if properties is not None:
        vehicle_properties.update(properties)
    return optimizer.add_vehicle(
        Event(
            "taxi-fleet",
            "added-taxi",
            {
                "id": vehicle_id,
                "intersection-id": intersection_id,
                "properties": vehicle_properties,
            },
        )
    )


def ride_request(
    request_id: str = "request-1",
    start_intersection_id: int = 2,
    end_intersection_id: int = 3,
    customers: int = 1,
    earliest_service_time: int | None = None,
    latest_service_time: int | None = None,
) -> Event:
    data = {
        "id": request_id,
        "start-intersection-id": start_intersection_id,
        "end-intersection-id": end_intersection_id,
        "number-of-customers": customers,
    }
    if earliest_service_time is not None:
        data["earliest-service-time"] = earliest_service_time
    if latest_service_time is not None:
        data["latest-service-time"] = latest_service_time
    return Event(
        "request",
        "ride-request-received",
        data,
    )


def test_plan_route_assigns_lowest_travel_vehicle() -> None:
    optimizer = MasterEvolutionaryLoop(build_world(), settings())
    add_vehicle(optimizer, "vehicle-1", 1)
    add_vehicle(optimizer, "vehicle-2", 2)

    responses = optimizer.plan_route(ride_request())

    assert len(responses) == 1
    assert responses[0].category == "taxi-fleet"
    assert responses[0].name == "plan-route"
    assert responses[0].get("vehicle-id") == "vehicle-2"
    assert responses[0].get("request-id") == "request-1"
    assert responses[0].explanations["taxi-manager"].startswith("Selected vehicle-2")
    assert responses[0].get("route") == [
        {
            "type": "pick-up-passengers",
            "request-id": "request-1",
            "intersection-id": 2,
            "count": 1,
        },
        {"type": "follow-road", "road-id": 20},
        {
            "type": "drop-off-passengers",
            "request-id": "request-1",
            "intersection-id": 3,
            "count": 1,
        },
    ]
    assert optimizer.vehicles["vehicle-2"].is_busy is True
    assert optimizer.open_requests == {}


def test_multi_customer_request_uses_available_vehicle_capacity() -> None:
    optimizer = MasterEvolutionaryLoop(build_world(), settings())
    add_vehicle(optimizer, "vehicle-1", 1, capacity=2)
    add_vehicle(optimizer, "vehicle-2", 1, capacity=2)

    responses = optimizer.plan_route(ride_request(start_intersection_id=1, customers=3))

    assert len(responses) == 2
    pickup_counts = [
        step["count"]
        for response in responses
        for step in response.get("route")
        if step["type"] == "pick-up-passengers"
    ]
    assert sorted(pickup_counts) == [1, 2]
    assert optimizer.open_requests == {}
    assert sum(vehicle.is_busy for vehicle in optimizer.vehicles.values()) == 2


def test_open_request_is_assigned_when_vehicle_finishes_move() -> None:
    optimizer = MasterEvolutionaryLoop(build_world(), settings())
    add_vehicle(optimizer, "vehicle-1", 1, capacity=1)

    first_responses = optimizer.plan_route(ride_request(start_intersection_id=1, customers=2))
    second_responses = optimizer.vehicle_available(
        Event("vehicle", "finished-move", {"vehicle-id": "vehicle-1"})
    )

    assert len(first_responses) == 1
    assert len(second_responses) == 1
    assert second_responses[0].get("vehicle-id") == "vehicle-1"
    assert optimizer.open_requests == {}


def test_evolutionary_planner_can_choose_cleaner_non_shortest_route() -> None:
    world = WorldManager("http://localhost:8088")
    world.add_intersections([{"id": 1}, {"id": 2}, {"id": 3}])
    world.add_roads(
        _roads_with_return_edges(
            [
                {"id": 10, "start-node": 1, "end-node": 2, "length": 1000.0, "maximum-speed": 100},
                {"id": 20, "start-node": 1, "end-node": 3, "length": 500.0, "maximum-speed": 20},
                {"id": 30, "start-node": 3, "end-node": 2, "length": 500.0, "maximum-speed": 20},
            ]
        )
    )
    optimizer = MasterEvolutionaryLoop(
        world,
        EvolutionarySettings(
            population_size=32,
            generations=20,
            seed=5,
            route_candidates_per_leg=2,
            time_weight=0.0,
            cost_weight=0.0,
            co2_weight=1.0,
        ),
    )
    add_vehicle(
        optimizer,
        "vehicle-1",
        1,
        capacity=1,
        properties={
            "maximum-speed": 100,
            "mass": 1000.0,
            "energy-efficiency-constant": 1.0,
            "friction-constant": 0.1,
            "resistance-constant": 1.0e-4,
            "co2-factor": 1.0,
        },
    )

    responses = optimizer.plan_route(ride_request(start_intersection_id=1, end_intersection_id=2))

    assert len(responses) == 1
    assert responses[0].get("route") == [
        {
            "type": "pick-up-passengers",
            "request-id": "request-1",
            "intersection-id": 1,
            "count": 1,
        },
        {"type": "follow-road", "road-id": 20},
        {"type": "follow-road", "road-id": 30},
        {
            "type": "drop-off-passengers",
            "request-id": "request-1",
            "intersection-id": 2,
            "count": 1,
        },
    ]


def test_late_service_time_is_penalized() -> None:
    optimizer = MasterEvolutionaryLoop(build_world(), settings())
    optimizer.update_time(Event("time", "time", {"time": 100}))
    add_vehicle(optimizer, "vehicle-1", 1, capacity=1)

    responses = optimizer.plan_route(
        ride_request(
            start_intersection_id=1,
            end_intersection_id=3,
            customers=1,
            latest_service_time=101,
        )
    )

    assert len(responses) == 1
    assert optimizer.knowledge.last_objectives is not None
    assert optimizer.knowledge.last_objectives[1] > 100


def test_planner_prefers_faster_vehicle_for_tight_deadline() -> None:
    world = WorldManager("http://localhost:8088")
    world.add_intersections([{"id": 1}, {"id": 2}])
    world.add_roads(
        _roads_with_return_edges(
            [{"id": 10, "start-node": 1, "end-node": 2, "length": 2000.0, "maximum-speed": 200}]
        )
    )
    optimizer = MasterEvolutionaryLoop(
        world,
        EvolutionarySettings(
            population_size=8,
            generations=4,
            seed=11,
            replan_debounce_seconds=0,
            enable_idle_repositioning=False,
        ),
    )
    add_vehicle(optimizer, "taxi-fast", 1, capacity=1, properties={"maximum-speed": 200})
    add_vehicle(optimizer, "taxi-slow", 1, capacity=1, properties={"maximum-speed": 20})

    responses = optimizer.plan_route(
        ride_request(start_intersection_id=1, end_intersection_id=2, latest_service_time=100)
    )

    assert len(responses) == 1
    assert responses[0].get("vehicle-id") == "taxi-fast"


def test_idle_clustered_taxi_is_repositioned() -> None:
    world = WorldManager("http://localhost:8088")
    world.add_intersections(
        [
            {"id": 1, "latitude": 53.0, "longitude": 8.0},
            {"id": 2, "latitude": 53.0005, "longitude": 8.0005},
            {"id": 3, "latitude": 53.05, "longitude": 8.05},
        ]
    )
    world.add_roads(
        _roads_with_return_edges(
            [
                {"id": 12, "start-node": 1, "end-node": 2, "length": 500.0, "maximum-speed": 200},
                {"id": 23, "start-node": 2, "end-node": 3, "length": 2000.0, "maximum-speed": 200},
            ]
        )
    )
    optimizer = MasterEvolutionaryLoop(
        world,
        EvolutionarySettings(
            replan_debounce_seconds=0,
            enable_idle_repositioning=True,
            idle_min_spacing_meters=1500.0,
        ),
    )
    add_vehicle(optimizer, "taxi-a", 1)
    responses = add_vehicle(optimizer, "taxi-b", 2)

    reposition = [
        response
        for response in responses
        if response.name == "plan-route" and response.get("vehicle-id") == "taxi-b"
    ]
    assert len(reposition) == 1
    assert reposition[0].get("route")


def test_infeasible_request_remains_open() -> None:
    world = WorldManager("http://localhost:8088")
    world.add_intersections([{"id": 1}, {"id": 2}])
    optimizer = MasterEvolutionaryLoop(world, settings())
    add_vehicle(optimizer, "vehicle-1", 1)

    responses = optimizer.plan_route(
        ride_request(start_intersection_id=1, end_intersection_id=2)
    )

    assert responses == []
    assert list(optimizer.open_requests) == ["request-1"]


def test_reset_clears_optimizer_and_world_state() -> None:
    world = build_world()
    optimizer = MasterEvolutionaryLoop(world, settings())
    add_vehicle(optimizer, "vehicle-1", 1, capacity=1)
    optimizer.plan_route(ride_request(start_intersection_id=1, customers=2))

    assert optimizer.knowledge.cached_chromosomes

    optimizer.on_reset(Event("simulation", "reset"))

    assert optimizer.vehicles == {}
    assert optimizer.open_requests == {}
    assert optimizer.knowledge.cached_chromosomes == []
    assert optimizer.knowledge.best_chromosome is None
    assert optimizer.knowledge.last_reoptimization_time is None
    assert world.intersections == {}
    assert world.roads == {}


def test_optimizer_caches_chromosomes_after_successful_adaptation() -> None:
    optimizer = MasterEvolutionaryLoop(build_world(), fast_settings(warm_start_cache_size=3))
    add_vehicle(optimizer, "vehicle-1", 1)

    responses = optimizer.plan_route(ride_request(start_intersection_id=1))

    assert len(responses) == 1
    assert optimizer.knowledge.best_chromosome is not None
    assert len(optimizer.knowledge.cached_chromosomes) <= 3
    assert optimizer.knowledge.cached_chromosomes


def test_cached_chromosome_is_used_as_initial_seed() -> None:
    optimizer = MasterEvolutionaryLoop(build_world(), fast_settings())
    add_vehicle(optimizer, "vehicle-1", 1, capacity=1)
    optimizer.dispatch_monitor.record_request(ride_request(start_intersection_id=1))
    snapshot = optimizer.analyze.build_problem()
    cached = CachedChromosome.from_snapshot(np.array([1]), snapshot)
    optimizer.knowledge.cached_chromosomes = [cached]

    seeds = optimizer._planner._initial_chromosomes(snapshot)

    assert any(seed.tolist() == [1] for seed in seeds)


def test_cached_chromosome_is_repaired_when_vehicle_order_changes() -> None:
    first_optimizer = MasterEvolutionaryLoop(build_world(), fast_settings())
    add_vehicle(first_optimizer, "vehicle-1", 1, capacity=1)
    first_optimizer.dispatch_monitor.record_request(ride_request(start_intersection_id=1))
    first_snapshot = first_optimizer.analyze.build_problem()
    cached = CachedChromosome.from_snapshot(np.array([1]), first_snapshot)

    second_optimizer = MasterEvolutionaryLoop(build_world(), fast_settings())
    add_vehicle(second_optimizer, "vehicle-2", 2, capacity=1)
    add_vehicle(second_optimizer, "vehicle-1", 1, capacity=1)
    second_optimizer.dispatch_monitor.record_request(ride_request(start_intersection_id=1))
    second_snapshot = second_optimizer.analyze.build_problem()

    assert second_snapshot.repair_cached_chromosome(cached).tolist() == [0, 1]


def test_time_events_trigger_throttled_reoptimization_for_open_requests() -> None:
    optimizer = MasterEvolutionaryLoop(build_world(), fast_settings())
    add_vehicle(optimizer, "vehicle-1", 1, capacity=1)
    optimizer.plan_route(ride_request(start_intersection_id=1, customers=2))
    optimizer.vehicles["vehicle-1"].is_busy = False

    assert optimizer.update_time(Event("time", "time", {"time": 4})) == []

    responses = optimizer.update_time(Event("time", "time", {"time": 5}))

    assert len(responses) == 1
    assert optimizer.open_requests == {}


def test_road_network_changes_trigger_reoptimization() -> None:
    world = build_world()
    optimizer = MasterEvolutionaryLoop(world, fast_settings())
    dispatcher = EventDispatcher(world, optimizer)
    add_vehicle(optimizer, "vehicle-1", 1, capacity=1)
    optimizer.dispatch_monitor.record_request(ride_request(start_intersection_id=1))

    responses = dispatcher.handle_event(
        Event(
            "road-network",
            "changed-road-property",
            {"road-id": 10, "properties": {"maximum-speed": 50}},
        )
    )

    assert len(responses) == 1
    assert responses[0].get("vehicle-id") == "vehicle-1"


def test_master_dispatch_owns_analyze_and_plan_taxis_are_slaves() -> None:
    optimizer = MasterMapeKLoop(build_world(), settings())
    add_vehicle(optimizer, "vehicle-1", 1)

    assert hasattr(optimizer, "analyze")
    assert hasattr(optimizer, "dispatch_monitor")
    assert callable(optimizer.plan)
    assert optimizer.knowledge is optimizer.global_knowledge
    assert not hasattr(optimizer, "monitor")
    assert not hasattr(optimizer, "execute")
    assert "vehicle-1" in optimizer.taxi_slaves
    taxi = optimizer.taxi_slaves["vehicle-1"]
    assert taxi.vehicle_id == "vehicle-1"
    assert hasattr(taxi, "monitor")
    assert hasattr(taxi, "execute")
    assert hasattr(taxi, "local")
    assert taxi.local.vehicle_id == "vehicle-1"


def test_local_knowledge_records_monitor_and_execute_context() -> None:
    optimizer = MasterEvolutionaryLoop(build_world(), settings())
    add_vehicle(optimizer, "vehicle-1", 1)
    add_vehicle(optimizer, "vehicle-2", 2)

    responses = optimizer.plan_route(ride_request())
    local = optimizer.taxi_slaves["vehicle-2"].local

    assert len(responses) == 1
    assert local.last_monitor_operation == "record_vehicle"
    assert len(local.last_executed_decisions) == 1
    assert local.last_executed_decisions[0]["vehicle_id"] == "vehicle-2"
    assert len(local.active_route) > 0


def test_distributed_local_knowledge_is_tracked_on_master() -> None:
    with MasterMapeKLoop(build_world(), settings(), distributed=True) as optimizer:
        add_vehicle(optimizer, "vehicle-1", 1)
        optimizer.plan_route(ride_request())
        local = optimizer.local_knowledge_for("vehicle-1")

        assert local is not None
        assert local.vehicle_id == "vehicle-1"
        assert len(local.last_executed_decisions) == 1


def test_plan_hook_can_be_called_directly() -> None:
    optimizer = MasterEvolutionaryLoop(build_world(), settings())
    add_vehicle(optimizer, "vehicle-1", 1)
    optimizer.dispatch_monitor.record_request(ride_request())

    result = optimizer.plan()

    assert len(result.decisions) == 1
    assert result.served_customers == 1
    assert optimizer.knowledge.last_objectives is not None


def test_distributed_taxi_slaves_run_in_separate_processes() -> None:
    with MasterMapeKLoop(build_world(), settings(), distributed=True) as optimizer:
        assert optimizer.distributed is True
        assert optimizer._taxi_pool is not None

        add_vehicle(optimizer, "vehicle-1", 1)
        add_vehicle(optimizer, "vehicle-2", 2)
        assert optimizer._taxi_pool.has_taxi("vehicle-1")
        assert optimizer._taxi_pool.has_taxi("vehicle-2")
        assert optimizer._taxi_pool.is_alive

        responses = optimizer.plan_route(ride_request())

    assert len(responses) == 1
    assert responses[0].get("vehicle-id") == "vehicle-2"


def test_one_taxi_slave_is_created_per_vehicle() -> None:
    optimizer = MasterMapeKLoop(build_world(), settings())
    add_vehicle(optimizer, "vehicle-1", 1)
    add_vehicle(optimizer, "vehicle-2", 2)

    assert optimizer.slave_count == 2
    assert set(optimizer.taxi_slaves) == {"vehicle-1", "vehicle-2"}


def test_custom_tailored_explanations() -> None:
                                                        
    custom_settings = EvolutionarySettings(
        population_size=4,
        generations=1,
        seed=11,
        time_weight=0.4,
        co2_weight=0.4,
        throughput_weight=0.2,
    )
    optimizer = MasterEvolutionaryLoop(build_world(), custom_settings)

                          
    add_vehicle(
        optimizer,
        "taxi-electric",
        2,
        properties={
            "motor-type": "electric",
            "maximum-speed": 100,
            "mass": 1000.0,
            "energy-efficiency-constant": 0.8,
            "co2-factor": 110.0,
            "resistance-constant": 1.7e-4,
            "friction-constant": 0.08,
        }
    )

                                                                              
    add_vehicle(
        optimizer,
        "taxi-combustion",
        1,
        properties={
            "motor-type": "combustion-engine",
            "maximum-speed": 80,
            "mass": 1200.0,
            "energy-efficiency-constant": 0.25,
            "co2-factor": 260.0,
            "resistance-constant": 2.9e-4,
            "friction-constant": 0.09,
        }
    )

    responses = optimizer.plan_route(ride_request(start_intersection_id=2, end_intersection_id=3))

    assert len(responses) == 1
    resp = responses[0]
    assert resp.get("vehicle-id") == "taxi-electric"

    explanations = resp.explanations
    assert "customer" in explanations
    assert "taxi-driver" in explanations
    assert "taxi-manager" in explanations

                                                                                    
    assert "taxi-electric" in explanations["customer"]
    assert "electric taxi" in explanations["customer"]
    assert "carbon footprint" in explanations["customer"]

                                                             
    assert "Proceed to intersection 2" in explanations["taxi-driver"]
    assert "navigate to intersection 3" in explanations["taxi-driver"]
    assert "pick up 1 passenger" in explanations["taxi-driver"]

                                                                        
    assert "Goal weights: Time=0.40, CO2=0.40, Throughput=0.20" in explanations["taxi-manager"]
    assert "electric (Non-ICE)" in explanations["taxi-manager"]


def test_combustion_vehicle_explanation() -> None:
    custom_settings = EvolutionarySettings(
        population_size=4,
        generations=1,
        seed=11,
        time_weight=0.5,
        co2_weight=0.0,
        throughput_weight=0.5,
    )
    optimizer = MasterEvolutionaryLoop(build_world(), custom_settings)

                                
    add_vehicle(
        optimizer,
        "taxi-combustion",
        2,
        properties={
            "motor-type": "combustion-engine",
            "maximum-speed": 80,
            "mass": 1200.0,
            "energy-efficiency-constant": 0.25,
            "co2-factor": 260.0,
            "resistance-constant": 2.9e-4,
            "friction-constant": 0.09,
        }
    )

    responses = optimizer.plan_route(ride_request(start_intersection_id=2, end_intersection_id=3))

    assert len(responses) == 1
    resp = responses[0]
    assert resp.get("vehicle-id") == "taxi-combustion"

    explanations = resp.explanations
    assert "combustion-engine taxi" in explanations["customer"]
    assert "combustion (ICE)" in explanations["taxi-manager"]


def test_trivial_problem_skips_nsga_and_assigns_request() -> None:
    optimizer = MasterEvolutionaryLoop(
        build_world(),
        fast_settings(trivial_search_threshold=64),
    )
    add_vehicle(optimizer, "vehicle-1", 1, capacity=1)
    optimizer.dispatch_monitor.record_request(ride_request(start_intersection_id=1))

    result = optimizer.plan()

    assert len(result.decisions) == 1
    assert result.decisions[0].vehicle.id == "vehicle-1"


def test_replan_debounce_skips_non_critical_adaptation() -> None:
    optimizer = MasterEvolutionaryLoop(
        build_world(),
        fast_settings(replan_debounce_seconds=5, continuous_reoptimization_interval=0),
    )
    add_vehicle(optimizer, "vehicle-1", 1, capacity=1)
    add_vehicle(optimizer, "vehicle-2", 2, capacity=1)
    optimizer.dispatch_monitor.record_time(Event("time", "time", {"time": 1}))

    debounced = optimizer.add_vehicle(
        Event(
            "taxi-fleet",
            "added-taxi",
            {
                "id": "vehicle-3",
                "intersection-id": 3,
                "properties": {"maximum-capacity": 1},
            },
        )
    )
    forced = optimizer.plan_route(ride_request(request_id="request-2", start_intersection_id=1))

    assert debounced == []
    assert len(forced) == 1


def test_vehicle_available_assigns_remaining_customers_despite_debounce() -> None:
    optimizer = MasterEvolutionaryLoop(
        build_world(),
        EvolutionarySettings(replan_debounce_seconds=2),
    )
    add_vehicle(optimizer, "vehicle-1", 1, capacity=1)
    first = optimizer.plan_route(ride_request(start_intersection_id=2, customers=2))
    optimizer.vehicles["vehicle-1"].is_busy = False

    second = optimizer.vehicle_available(
        Event("vehicle", "finished-move", {"vehicle-id": "vehicle-1"})
    )

    assert len(first) == 1
    assert len(second) == 1
    assert optimizer.open_requests == {}


def test_default_dispatch_uses_evolutionary_path() -> None:
    optimizer = MasterEvolutionaryLoop(build_world(), EvolutionarySettings())
    add_vehicle(optimizer, "vehicle-1", 1, capacity=1)

    responses = optimizer.plan_route(ride_request(start_intersection_id=1, end_intersection_id=3))

    assert len(responses) == 1
    assert responses[0].get("vehicle-id") == "vehicle-1"
    assert optimizer.open_requests == {}
    assert optimizer.knowledge.best_chromosome is not None


def test_duplicate_plan_route_event_does_not_dispatch_second_taxi() -> None:
    optimizer = MasterEvolutionaryLoop(build_world(), fast_settings())
    add_vehicle(optimizer, "vehicle-1", 1, capacity=1)
    add_vehicle(optimizer, "vehicle-2", 2, capacity=1)
    request = ride_request(start_intersection_id=1, end_intersection_id=3)

    first = optimizer.plan_route(request)
    second = optimizer.plan_route(request)

    assert len(first) == 1
    assert second == []
    assert optimizer.knowledge.request_is_committed("request-1")


def test_cancelled_request_stops_assigned_vehicle() -> None:
    optimizer = MasterEvolutionaryLoop(build_world(), fast_settings())
    add_vehicle(optimizer, "vehicle-1", 1, capacity=1)
    optimizer.plan_route(ride_request(start_intersection_id=1, end_intersection_id=3))

    responses = optimizer.cancel_request(
        Event("request", "customer-not-served", {"request-id": "request-1"})
    )

    assert len(responses) == 1
    assert responses[0].category == "taxi-fleet"
    assert responses[0].name == "plan-route"
    assert responses[0].get("vehicle-id") == "vehicle-1"
    assert responses[0].get("route") == []
    assert optimizer.knowledge.request_is_cancelled("request-1")
    assert optimizer.vehicles["vehicle-1"].is_busy is False
    assert optimizer.open_requests == {}


def test_cancelled_request_is_not_reassigned_when_vehicle_becomes_available() -> None:
    optimizer = MasterEvolutionaryLoop(build_world(), fast_settings())
    add_vehicle(optimizer, "vehicle-1", 1, capacity=1)
    optimizer.plan_route(ride_request(request_id="request-1", start_intersection_id=1))
    optimizer.cancel_request(
        Event("request", "customer-not-served", {"request-id": "request-1"})
    )
    optimizer.vehicles["vehicle-1"].is_busy = False

    responses = optimizer.vehicle_available(
        Event("vehicle", "finished-move", {"vehicle-id": "vehicle-1"})
    )

    assert responses == []


def build_assignment_decision(
    optimizer: MasterEvolutionaryLoop,
    vehicle_id: str,
    request,
) -> AssignmentDecision:
    vehicle = optimizer.vehicles[vehicle_id]
    world = optimizer.global_knowledge.world
    to_pickup = world.shortest_path_route(
        vehicle.position.id,
        request.start_intersection.id,
        vehicle.properties,
    )
    to_destination = world.shortest_path_route(
        request.start_intersection.id,
        request.end_intersection.id,
        vehicle.properties,
    )
    return AssignmentDecision(
        vehicle=vehicle,
        request=request,
        to_pickup=to_pickup,
        to_destination=to_destination,
        served_customers=1,
        travel_time=to_pickup.travel_time + to_destination.travel_time,
        distance=to_pickup.distance + to_destination.distance,
        co2_emissions=to_pickup.co2_emissions + to_destination.co2_emissions,
        distance_cost=to_pickup.distance_cost + to_destination.distance_cost,
    )


def test_execute_decisions_does_not_double_assign_same_request() -> None:
    optimizer = MasterEvolutionaryLoop(build_world(), settings())
    add_vehicle(optimizer, "vehicle-1", 1, capacity=1)
    add_vehicle(optimizer, "vehicle-2", 1, capacity=1)
    optimizer.dispatch_monitor.record_request(
        ride_request(start_intersection_id=1, end_intersection_id=3, customers=1)
    )
    request = optimizer.open_requests["request-1"]

    first = build_assignment_decision(optimizer, "vehicle-1", request)
    second = build_assignment_decision(optimizer, "vehicle-2", request)
    assert first is not None and second is not None

    responses = optimizer._execute_decisions([first, second])

    assert len(responses) == 1
    assert sum(vehicle.is_busy for vehicle in optimizer.vehicles.values()) == 1
    assert optimizer.open_requests == {}


def test_ride_request_expired_event_stops_assigned_vehicle() -> None:
    optimizer = MasterEvolutionaryLoop(build_world(), fast_settings())
    add_vehicle(optimizer, "vehicle-1", 1, capacity=1)
    optimizer.plan_route(ride_request(start_intersection_id=1, end_intersection_id=3))

    responses = optimizer.cancel_request(
        Event("request", "ride-request-expired", {"id": "request-1"})
    )

    assert len(responses) == 1
    assert responses[0].get("route") == []
    assert responses[0].get("vehicle-id") == "vehicle-1"
    assert optimizer.knowledge.request_is_cancelled("request-1")
    assert optimizer.vehicles["vehicle-1"].is_busy is False


def test_expired_request_stops_assigned_vehicle_on_time_update() -> None:
    optimizer = MasterEvolutionaryLoop(build_world(), fast_settings())
    add_vehicle(optimizer, "vehicle-1", 1, capacity=1)
    optimizer.plan_route(
        ride_request(
            start_intersection_id=1,
            end_intersection_id=3,
            latest_service_time=100,
        )
    )

    responses = optimizer.update_time(Event("time", "time", {"time": 101}))

    assert any(
        response.category == "taxi-fleet"
        and response.name == "plan-route"
        and response.get("vehicle-id") == "vehicle-1"
        and response.get("route") == []
        for response in responses
    )
    assert optimizer.knowledge.request_is_cancelled("request-1")
    assert optimizer.vehicles["vehicle-1"].is_busy is False


def test_already_expired_request_is_not_dispatched() -> None:
    optimizer = MasterEvolutionaryLoop(build_world(), fast_settings())
    add_vehicle(optimizer, "vehicle-1", 1, capacity=1)
    optimizer.dispatch_monitor.record_time(Event("time", "time", {"time": 200}))

    responses = optimizer.plan_route(
        ride_request(
            start_intersection_id=1,
            end_intersection_id=3,
            latest_service_time=100,
        )
    )

    assert responses == []
    assert optimizer.knowledge.request_is_cancelled("request-1")
    assert optimizer.vehicles["vehicle-1"].is_busy is False


def test_picked_up_request_is_not_cancelled_on_expiry() -> None:
    optimizer = MasterEvolutionaryLoop(build_world(), fast_settings())
    add_vehicle(optimizer, "vehicle-1", 1, capacity=1)
    optimizer.plan_route(
        ride_request(
            start_intersection_id=1,
            end_intersection_id=3,
            latest_service_time=100,
        )
    )
    optimizer.record_pickup(
        Event(
            "taxi-fleet",
            "picked-up-passengers",
            {"request-id": "request-1", "vehicle-id": "vehicle-1"},
        )
    )

    responses = optimizer.update_time(Event("time", "time", {"time": 101}))

    assert not any(response.get("route") == [] for response in responses)
    assert not optimizer.knowledge.request_is_cancelled("request-1")
    assert optimizer.vehicles["vehicle-1"].is_busy is True


def test_person_removed_does_not_cancel_active_dispatch() -> None:
    optimizer = MasterEvolutionaryLoop(build_world(), fast_settings())
    add_vehicle(optimizer, "vehicle-1", 1, capacity=1)
    optimizer.record_person(
        Event("person", "added", {"id": "person-1", "request-id": "request-1"})
    )
    optimizer.plan_route(ride_request(request_id="request-1", start_intersection_id=1))

    responses = optimizer.cancel_request(
        Event("person", "removed", {"id": "person-1"})
    )

    assert responses == []
    assert not optimizer.knowledge.request_is_cancelled("request-1")
    assert optimizer.knowledge.request_is_committed("request-1")


def test_plan_reuses_cached_evaluations(monkeypatch) -> None:
    optimizer = MasterEvolutionaryLoop(build_world(), settings())
    add_vehicle(optimizer, "vehicle-1", 1, capacity=1)
    add_vehicle(optimizer, "vehicle-2", 2, capacity=1)
    snapshot = optimizer.analyze.build_problem()
    evaluation_calls = {"count": 0}
    original_evaluate = optimizer._planner.evaluators[0].evaluate

    def counting_evaluate(chromosome, problem_snapshot):
        evaluation_calls["count"] += 1
        return original_evaluate(chromosome, problem_snapshot)

    monkeypatch.setattr(optimizer._planner.evaluators[0], "evaluate", counting_evaluate)

    optimizer._planner.plan(snapshot)

    assert evaluation_calls["count"] < settings().population_size * settings().generations
