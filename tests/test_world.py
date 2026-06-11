from src.events import Event
from src.world import WorldManager


def _roads_with_return_edges(roads: list[dict]) -> list[dict]:

    nodes = {int(road["start-node"]) for road in roads} | {int(road["end-node"]) for road in roads}
    next_id = max(int(road["id"]) for road in roads) + 1
    extra = []
    for node_id in sorted(nodes):
        has_outgoing = any(int(road["start-node"]) == node_id for road in roads)
        if not has_outgoing:
            target = 1 if node_id != 1 else next((n for n in nodes if n != node_id), node_id)
            extra.append(
                {
                    "id": next_id,
                    "start-node": node_id,
                    "end-node": target,
                    "length": 100.0,
                    "maximum-speed": 60,
                }
            )
            next_id += 1
    return [*roads, *extra]


def test_shortest_path_uses_road_duration() -> None:
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

    path = world.shortest_path(1, 3)

    assert path is not None
    assert [road.id for road in path.edge_list] == [10, 20]
    assert path.weight == 118


def test_road_property_change_updates_routing_weight() -> None:
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

    world.change_road_properties(
        Event("road-network", "changed-road-property", {"road-id": 30, "properties": {"maximum-speed": 240}})
    )
    path = world.shortest_path(1, 3)

    assert path is not None
    assert [road.id for road in path.edge_list] == [30]


def test_route_candidates_include_alternative_paths() -> None:
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

    candidates = world.route_candidates(1, 2, {}, max_candidates=2, search_limit=4)

    assert [candidate.road_ids for candidate in candidates] == [(10,), (20, 30)]


def test_vehicle_speed_affects_route_duration() -> None:
    world = WorldManager("http://localhost:8088")
    world.add_intersections([{"id": 1}, {"id": 2}])
    world.add_roads(
        _roads_with_return_edges(
            [{"id": 10, "start-node": 1, "end-node": 2, "length": 1000.0, "maximum-speed": 100}]
        )
    )

    fast = world.route_candidates(1, 2, {"maximum-speed": 100}, max_candidates=1, search_limit=1)[0]
    slow = world.route_candidates(1, 2, {"maximum-speed": 50}, max_candidates=1, search_limit=1)[0]

    assert fast.travel_time == 36
    assert slow.travel_time == 72


def test_route_metrics_include_distance_cost_and_co2() -> None:
    world = WorldManager("http://localhost:8088")
    world.add_intersections([{"id": 1}, {"id": 2}])
    world.add_roads(
        _roads_with_return_edges(
            [{"id": 10, "start-node": 1, "end-node": 2, "length": 1000.0, "maximum-speed": 100}]
        )
    )
    vehicle_properties = {
        "maximum-speed": 100,
        "mass": 1000.0,
        "energy-efficiency-constant": 1.0,
        "friction-constant": 0.1,
        "resistance-constant": 1.0e-4,
        "co2-factor": 1.0,
        "cost-per-meter": 0.05,
        "distance-cost-factor": 2.0,
    }

    candidate = world.route_candidates(1, 2, vehicle_properties, max_candidates=1, search_limit=1)[0]

    assert candidate.distance == 1000.0
    assert candidate.distance_cost == 100.0
    assert candidate.co2_emissions > 0.0


def test_ensure_road_network_loaded_fetches_when_cache_is_empty(monkeypatch) -> None:
    world = WorldManager("http://localhost:8088")

    def fake_get(url: str, timeout: int = 30):
        assert timeout == 30
        response = type("Response", (), {"raise_for_status": lambda self: None})()
        if url.endswith("/intersections"):
            response.json = lambda: [{"id": 1}, {"id": 2}]
        else:
            response.json = lambda: [
                {
                    "id": 10,
                    "start-node": 1,
                    "end-node": 2,
                    "length": 1000.0,
                    "maximum-speed": 60,
                }
            ]
        return response

    monkeypatch.setattr("src.world.requests.get", fake_get)

    assert world.ensure_road_network_loaded() is True
    assert world.get_intersection(1) is not None
    assert world.ensure_road_network_loaded() is True


def test_fetch_road_network_returns_initialized_event(monkeypatch) -> None:
    world = WorldManager("http://localhost:8088")
    monkeypatch.setattr(world, "_load_road_network", lambda: True)

    responses = world.fetch_road_network(Event("simulation", "initialize"))

    assert len(responses) == 1
    assert responses[0].category == "client"
    assert responses[0].name == "initialized"


def test_reset_clears_road_network() -> None:
    world = WorldManager("http://localhost:8088")
    world.add_intersections([{"id": 1}, {"id": 2}])
    world.add_roads(
        _roads_with_return_edges(
            [{"id": 10, "start-node": 1, "end-node": 2, "length": 1000.0, "maximum-speed": 60}]
        )
    )

    world.on_reset()

    assert world.intersections == {}
    assert world.roads == {}
    assert world.shortest_path(1, 2) is None


def test_shortest_path_with_banned_road() -> None:
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

    alternate = world._shortest_path(1, 3, {}, {10})

    assert alternate is not None
    assert [road.id for road in alternate.edge_list] == [30]


def test_route_candidate_cache_reuses_results() -> None:
    world = WorldManager("http://localhost:8088")
    world.add_intersections([{"id": 1}, {"id": 2}])
    world.add_roads(
        _roads_with_return_edges(
            [{"id": 10, "start-node": 1, "end-node": 2, "length": 1000.0, "maximum-speed": 100}]
        )
    )

    first = world.route_candidates(1, 2, {}, max_candidates=1, search_limit=1)
    assert len(world._route_candidate_cache) == 1
    second = world.route_candidates(1, 2, {}, max_candidates=1, search_limit=1)

    assert first == second
    assert len(world._route_candidate_cache) == 1


def test_route_candidate_cache_invalidated_on_road_change() -> None:
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

    before = world.route_candidates(1, 3, {}, max_candidates=1, search_limit=1)
    world.change_road_properties(
        Event("road-network", "changed-road-property", {"road-id": 30, "properties": {"maximum-speed": 240}})
    )
    after = world.route_candidates(1, 3, {}, max_candidates=1, search_limit=1)

    assert before[0].road_ids != after[0].road_ids


def test_route_candidates_with_single_limit_runs_one_shortest_path(monkeypatch) -> None:
    world = WorldManager("http://localhost:8088")
    intersections = [{"id": index} for index in range(1, 52)]
    roads = [
        {
            "id": 10 + index,
            "start-node": index,
            "end-node": index + 1,
            "length": 1000.0,
            "maximum-speed": 100,
        }
        for index in range(1, 51)
    ]
    world.add_intersections(intersections)
    world.add_roads(_roads_with_return_edges(roads))

    calls = {"count": 0}
    original = world._shortest_path

    def counting_shortest_path(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(world, "_shortest_path", counting_shortest_path)

    candidates = world.route_candidates(1, 51, {}, max_candidates=1, search_limit=1)

    assert len(candidates) == 1
    assert calls["count"] == 1


def test_route_candidates_respects_search_limit_on_long_paths(monkeypatch) -> None:
    world = WorldManager("http://localhost:8088")
    intersections = [{"id": index} for index in range(1, 52)]
    roads = [
        {
            "id": 10 + index,
            "start-node": index,
            "end-node": index + 1,
            "length": 1000.0,
            "maximum-speed": 100,
        }
        for index in range(1, 51)
    ]
    world.add_intersections(intersections)
    world.add_roads(_roads_with_return_edges(roads))

    calls = {"count": 0}
    original = world._shortest_path

    def counting_shortest_path(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(world, "_shortest_path", counting_shortest_path)

    candidates = world.route_candidates(1, 51, {}, max_candidates=2, search_limit=12)

    assert len(candidates) >= 1
    assert calls["count"] <= 13


def test_travel_times_from_returns_reachable_nodes() -> None:
    world = WorldManager("http://localhost:8088")
    world.add_intersections([{"id": 1}, {"id": 2}, {"id": 3}])
    world.add_roads(
        [
            {"id": 10, "start-node": 1, "end-node": 2, "length": 1000.0, "maximum-speed": 60},
            {"id": 20, "start-node": 2, "end-node": 3, "length": 1000.0, "maximum-speed": 60},
        ]
    )

    travel_times = world.travel_times_from(1, {})

    assert travel_times[1] == 0
    assert travel_times[2] == 59
    assert travel_times[3] == 118
    assert 4 not in travel_times


def test_shortest_path_rejects_sink_intersections() -> None:
    world = WorldManager("http://localhost:8088")
    world.add_intersections([{"id": 1}, {"id": 2}])
    world.add_roads(
        [{"id": 10, "start-node": 1, "end-node": 2, "length": 1000.0, "maximum-speed": 60}]
    )

    assert world.shortest_path(1, 2) is None
    assert world.shortest_path_route(1, 2, {}) is None


def test_route_steps_require_outgoing_at_all_nodes() -> None:
    world = WorldManager("http://localhost:8088")
    world.add_intersections([{"id": 1}, {"id": 2}, {"id": 3}])
    world.add_roads(
        _roads_with_return_edges(
            [
                {"id": 10, "start-node": 1, "end-node": 2, "length": 1000.0, "maximum-speed": 60},
                {"id": 20, "start-node": 2, "end-node": 3, "length": 1000.0, "maximum-speed": 60},
            ]
        )
    )

    valid_steps = [
        {"type": "follow-road", "road-id": 10},
        {"type": "pick-up-passengers", "request-id": "r1", "intersection-id": 2, "count": 1},
        {"type": "follow-road", "road-id": 20},
        {"type": "drop-off-passengers", "request-id": "r1", "intersection-id": 3, "count": 1},
    ]

    assert world.route_steps_have_outgoing_at_all_nodes(1, valid_steps)
