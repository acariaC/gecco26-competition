from src.request_validator import RequestValidator, ValidationResult
from src.world import WorldManager


def test_validate_request_with_dead_end_pickup():
    world = WorldManager("http://localhost:8088")
    world.add_intersections([{"id": 1}, {"id": 2}, {"id": 3}])
    world.add_roads([
        {"id": 10, "start-node": 1, "end-node": 2, "length": 1000.0, "maximum-speed": 60},
    ])

    validator = RequestValidator(world)
    validator.refresh_dead_ends()

    result = validator.validate_request("req-1", pickup_id=3, dropoff_id=2)

    assert result.valid is False
    assert result.reject_event is not None
    assert result.reject_event.category == "request"
    assert result.reject_event.name == "reject"
    assert result.reject_event.get("id") == "req-1"
    assert "dead-end" in result.reason


def test_validate_request_with_dead_end_dropoff():
    world = WorldManager("http://localhost:8088")
    world.add_intersections([{"id": 1}, {"id": 2}, {"id": 3}])
    world.add_roads([
        {"id": 10, "start-node": 1, "end-node": 2, "length": 1000.0, "maximum-speed": 60},
    ])

    validator = RequestValidator(world)
    validator.refresh_dead_ends()

    result = validator.validate_request("req-1", pickup_id=1, dropoff_id=3)

    assert result.valid is False
    assert result.reject_event is not None
    assert result.reject_event.category == "request"
    assert result.reject_event.name == "reject"
    assert result.reject_event.get("id") == "req-1"
    assert "dead-end" in result.reason


def test_validate_request_with_both_dead_ends():
    world = WorldManager("http://localhost:8088")
    world.add_intersections([{"id": 1}, {"id": 2}, {"id": 3}])
    world.add_roads([
        {"id": 10, "start-node": 1, "end-node": 2, "length": 1000.0, "maximum-speed": 60},
    ])

    validator = RequestValidator(world)
    validator.refresh_dead_ends()

    result = validator.validate_request("req-1", pickup_id=3, dropoff_id=3)

    assert result.valid is False
    assert result.reject_event is not None
    assert "pickup" in result.reason


def test_validate_request_with_valid_nodes():
    world = WorldManager("http://localhost:8088")
    world.add_intersections([{"id": 1}, {"id": 2}])
    world.add_roads([
        {"id": 10, "start-node": 1, "end-node": 2, "length": 1000.0, "maximum-speed": 60},
        {"id": 20, "start-node": 2, "end-node": 1, "length": 1000.0, "maximum-speed": 60},
    ])

    validator = RequestValidator(world)
    validator.refresh_dead_ends()

    result = validator.validate_request("req-1", pickup_id=1, dropoff_id=2)

    assert result.valid is True
    assert result.reject_event is None
    assert result.reason is None


def test_validate_request_with_no_dead_ends():
    world = WorldManager("http://localhost:8088")
    world.add_intersections([{"id": 1}, {"id": 2}, {"id": 3}])
    world.add_roads([
        {"id": 10, "start-node": 1, "end-node": 2, "length": 1000.0, "maximum-speed": 60},
        {"id": 20, "start-node": 2, "end-node": 3, "length": 1000.0, "maximum-speed": 60},
        {"id": 30, "start-node": 3, "end-node": 1, "length": 1000.0, "maximum-speed": 60},
    ])

    validator = RequestValidator(world)
    validator.refresh_dead_ends()

    assert validator.dead_end_count == 0

    result = validator.validate_request("req-1", pickup_id=1, dropoff_id=3)

    assert result.valid is True
    assert result.reject_event is None


def test_refresh_dead_ends_after_road_change():
    world = WorldManager("http://localhost:8088")
    world.add_intersections([{"id": 1}, {"id": 2}, {"id": 3}])
    world.add_roads([
        {"id": 10, "start-node": 1, "end-node": 2, "length": 1000.0, "maximum-speed": 60},
    ])

    validator = RequestValidator(world)
    validator.refresh_dead_ends()

    assert 2 in validator._dead_ends
    assert 3 in validator._dead_ends

    world.add_roads([
        {"id": 20, "start-node": 2, "end-node": 3, "length": 1000.0, "maximum-speed": 60},
    ])
    validator.refresh_dead_ends()

    assert 2 not in validator._dead_ends
    assert 3 in validator._dead_ends


def test_dead_end_count_property():
    world = WorldManager("http://localhost:8088")
    world.add_intersections([{"id": 1}, {"id": 2}, {"id": 3}])
    world.add_roads([
        {"id": 10, "start-node": 1, "end-node": 2, "length": 1000.0, "maximum-speed": 60},
    ])

    validator = RequestValidator(world)
    validator.refresh_dead_ends()

    assert validator.dead_end_count == 2


def test_reject_event_contains_explanation():
    world = WorldManager("http://localhost:8088")
    world.add_intersections([{"id": 1}, {"id": 2}])
    world.add_roads([
        {"id": 10, "start-node": 1, "end-node": 2, "length": 1000.0, "maximum-speed": 60},
    ])

    validator = RequestValidator(world)
    validator.refresh_dead_ends()

    result = validator.validate_request("req-1", pickup_id=2, dropoff_id=1)

    assert result.valid is False
    explanation = result.reject_event.get("explanation")
    assert explanation is not None
    assert "dead-end" in explanation["reason"]
