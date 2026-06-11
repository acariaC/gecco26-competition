from __future__ import annotations

import heapq
import itertools
import logging
import math
from dataclasses import dataclass, field
from typing import Any

import requests

from .events import Event

LOGGER = logging.getLogger(__name__)
ROAD_NETWORK_ATTRIBUTE_START_NODE = "start-node"
ROAD_NETWORK_ATTRIBUTE_END_NODE = "end-node"
ROAD_NETWORK_ATTRIBUTE_LENGTH = "length"
ROAD_NETWORK_ATTRIBUTE_MAX_SPEED = "maximum-speed"
ROAD_NETWORK_ATTRIBUTE_ID = "id"
ROUTING_PROPERTY_KEYS = (
    "maximum-speed",
    "mass",
    "energy-efficiency-constant",
    "energy-efficiency-factor",
    "friction-constant",
    "resistance-constant",
    "co2-factor",
    "cost-per-meter",
    "distance-cost-factor",
)


def routing_properties_key(properties: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple(
        (key, properties.get(key))
        for key in ROUTING_PROPERTY_KEYS
        if key in properties
    )


@dataclass(eq=False)
class Intersection:
    id: int
    properties: dict[str, Any]

    def __repr__(self) -> str:
        return f"Intersection<{self.id}>"


@dataclass(eq=False)
class Road:
    id: int
    length: float
    max_speed: float
    properties: dict[str, Any]

    def duration(self) -> float:
        return self.duration_for_vehicle({})

    def duration_for_vehicle(self, vehicle_properties: dict[str, Any]) -> float:
        speed = self.speed_for_vehicle(vehicle_properties)
        if speed <= 0:
            return float("inf")
        meters_per_second = 1000.0 / 3600.0
        return int(self.length / (speed * meters_per_second))

    def speed_for_vehicle(self, vehicle_properties: dict[str, Any]) -> float:
        vehicle_speed = float(vehicle_properties.get("maximum-speed", self.max_speed) or self.max_speed)
        return min(self.max_speed, vehicle_speed)

    def emissions_for_vehicle(self, vehicle_properties: dict[str, Any]) -> float:
        speed = self.speed_for_vehicle(vehicle_properties)
        if speed <= 0:
            return float("inf")

        mass = float(vehicle_properties.get("mass", 0.0) or 0.0)
        efficiency = float(
            vehicle_properties.get("energy-efficiency-constant")
            or vehicle_properties.get("energy-efficiency-factor")
            or 1.0
        )
        friction = float(vehicle_properties.get("friction-constant", 0.0) or 0.0)
        resistance = float(vehicle_properties.get("resistance-constant", 0.0) or 0.0)
        co2_factor = float(vehicle_properties.get("co2-factor", 0.0) or 0.0)

        if mass <= 0 or efficiency <= 0 or co2_factor <= 0:
            return 0.0

        vsp = speed * (friction + resistance * math.pow(speed, 3))
        energy = vsp * mass * (self.length / speed) / 3_600_000
        return (energy / efficiency) * co2_factor

    def distance_cost_for_vehicle(self, vehicle_properties: dict[str, Any]) -> float:
        cost_per_meter = float(vehicle_properties.get("cost-per-meter", 0.0) or 0.0)
        distance_factor = float(vehicle_properties.get("distance-cost-factor", 1.0) or 1.0)
        return self.length * cost_per_meter * distance_factor

    def to_follow_road_step(self) -> dict[str, Any]:
        return {"type": "follow-road", "road-id": self.id}


@dataclass
class Vehicle:
    id: str
    position: Intersection | None
    max_capacity: int
    properties: dict[str, Any]
    is_busy: bool = False
    last_active_time: int = 0


@dataclass
class RideRequest:
    earliest_service_time: int
    latest_service_time: int
    start_intersection: Intersection
    end_intersection: Intersection
    id: str
    number_of_customers: int
    remaining_customers: int = field(init=False)

    def __post_init__(self) -> None:
        self.remaining_customers = self.number_of_customers


@dataclass(frozen=True)
class GraphPath:
    edge_list: list[Road]
    weight: float


@dataclass(frozen=True)
class RouteCandidate:
    path: GraphPath
    distance: float
    travel_time: float
    co2_emissions: float
    distance_cost: float

    @property
    def road_ids(self) -> tuple[int, ...]:
        return tuple(road.id for road in self.path.edge_list)


class WorldManager:


    def __init__(self, http_url: str) -> None:
        self.url = http_url.rstrip("/") + "/simulation/road-network"
        self.intersections: dict[int, Intersection] = {}
        self.roads: dict[int, Road] = {}
        self._adjacency: dict[int, list[tuple[int, Road]]] = {}
        self._routing_revision = 0
        self._route_candidate_cache: dict[tuple[Any, ...], list[RouteCandidate]] = {}

    def get_intersection(self, intersection_id: int | float) -> Intersection | None:
        return self.intersections.get(int(intersection_id))

    def intersection_coords(self, intersection_id: int | float) -> tuple[float, float] | None:
        intersection = self.get_intersection(intersection_id)
        if intersection is None:
            return None
        latitude = intersection.properties.get("latitude")
        longitude = intersection.properties.get("longitude")
        if latitude is None or longitude is None:
            return None
        try:
            return float(latitude), float(longitude)
        except (TypeError, ValueError):
            return None

    def geo_distance(self, start_id: int | float, end_id: int | float) -> float | None:

        start = self.intersection_coords(start_id)
        end = self.intersection_coords(end_id)
        if start is None or end is None:
            return None

        earth_radius_meters = 6_371_000.0
        lat1, lon1 = math.radians(start[0]), math.radians(start[1])
        lat2, lon2 = math.radians(end[0]), math.radians(end[1])
        delta_lat = lat2 - lat1
        delta_lon = lon2 - lon1
        haversine = (
            math.sin(delta_lat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
        )
        return 2 * earth_radius_meters * math.asin(min(1.0, math.sqrt(haversine)))

    def ensure_road_network_loaded(self) -> bool:

        if self.intersections:
            return True
        return self._load_road_network()

    def fetch_road_network(self, event: Event) -> list[Event]:
        LOGGER.info("Fetching road network for event %s", event)
        if self._load_road_network():
            return [Event("client", "initialized")]
        return []

    def _load_road_network(self) -> bool:
        try:
            LOGGER.info("Loading road network from %s", self.url)
            intersections_response = requests.get(f"{self.url}/intersections", timeout=30)
            intersections_response.raise_for_status()
            self.add_intersections(intersections_response.json())

            roads_response = requests.get(f"{self.url}/roads", timeout=30)
            roads_response.raise_for_status()
            self.add_roads(roads_response.json())
        except requests.RequestException as error:
            LOGGER.error("Failed to load road network from %s: %s", self.url, error)
            return False

        LOGGER.info(
            "Loaded %s intersections and %s roads.",
            len(self.intersections),
            len(self.roads),
        )
        return bool(self.intersections)

    def change_intersection_properties(self, event: Event) -> list[Event]:
        intersection = self.get_intersection(event.get("intersection-id"))
        if intersection is not None:
            intersection.properties.update(dict(event.get("properties") or {}))
        return []

    def change_road_properties(self, event: Event) -> list[Event]:
        road_id = int(event.get("road-id"))
        road = self.roads.get(road_id)
        if road is not None:
            properties = dict(event.get("properties") or {})
            road.properties.update(properties)
            if ROAD_NETWORK_ATTRIBUTE_LENGTH in properties:
                road.length = float(properties[ROAD_NETWORK_ATTRIBUTE_LENGTH])
            if ROAD_NETWORK_ATTRIBUTE_MAX_SPEED in properties:
                road.max_speed = float(properties[ROAD_NETWORK_ATTRIBUTE_MAX_SPEED])
            self._invalidate_routing_cache()
        return []

    def on_reset(self) -> None:
        LOGGER.info("Resetting road network.")
        self.roads.clear()
        self.intersections.clear()
        self._adjacency.clear()
        self._invalidate_routing_cache()

    def add_intersections(self, intersections: list[dict[str, Any]] | None) -> None:
        if intersections is None:
            return

        for item in intersections:
            intersection_id = int(item[ROAD_NETWORK_ATTRIBUTE_ID])
            self.intersections[intersection_id] = Intersection(intersection_id, dict(item))
            self._adjacency.setdefault(intersection_id, [])

    def add_roads(self, roads: list[dict[str, Any]] | None) -> None:
        if roads is None:
            return

        for item in roads:
            road_id = int(item[ROAD_NETWORK_ATTRIBUTE_ID])
            road = Road(
                id=road_id,
                length=float(item[ROAD_NETWORK_ATTRIBUTE_LENGTH]),
                max_speed=float(item[ROAD_NETWORK_ATTRIBUTE_MAX_SPEED]),
                properties=dict(item),
            )
            start_id = int(item[ROAD_NETWORK_ATTRIBUTE_START_NODE])
            end_id = int(item[ROAD_NETWORK_ATTRIBUTE_END_NODE])

            if start_id not in self.intersections or end_id not in self.intersections:
                LOGGER.warning(
                    "Skipping road %s from %s to %s because an endpoint is unknown.",
                    road_id,
                    start_id,
                    end_id,
                )
                continue

            self.roads[road_id] = road
            self._adjacency.setdefault(start_id, []).append((end_id, road))
            self._adjacency.setdefault(end_id, self._adjacency.get(end_id, []))
            self._invalidate_routing_cache()

    def has_outgoing_edges(self, intersection_id: int | float) -> bool:

        return bool(self._adjacency.get(int(intersection_id), []))

    def dead_end_intersection_ids(self) -> set[int]:

        return {
            node_id
            for node_id in self.intersections
            if not self._adjacency.get(node_id)
        }

    def intersection_ids_on_graph_path(
        self,
        start_id: int | float,
        path: GraphPath,
    ) -> list[int] | None:

        current = int(start_id)
        if current not in self.intersections:
            return None

        nodes = [current]
        for road in path.edge_list:
            next_id = self._road_exit_node(current, road)
            if next_id is None:
                return None
            nodes.append(next_id)
            current = next_id
        return nodes

    def graph_path_has_outgoing_at_all_nodes(
        self,
        start_id: int | float,
        path: GraphPath,
    ) -> bool:

        nodes = self.intersection_ids_on_graph_path(start_id, path)
        if nodes is None:
            return False
        return all(self.has_outgoing_edges(node_id) for node_id in nodes)

    def intersection_ids_on_route_steps(
        self,
        start_id: int | float,
        steps: list[dict[str, Any]],
    ) -> list[int] | None:

        current = int(start_id)
        if current not in self.intersections:
            return None

        visited = [current]
        for step in steps:
            step_type = step.get("type")
            if step_type == "follow-road":
                road = self.roads.get(int(step["road-id"]))
                if road is None:
                    return None
                next_id = self._road_exit_node(current, road)
                if next_id is None:
                    return None
                current = next_id
                if visited[-1] != current:
                    visited.append(current)
            elif step_type in ("pick-up-passengers", "drop-off-passengers"):
                intersection_id = int(step["intersection-id"])
                current = intersection_id
                if visited[-1] != current:
                    visited.append(current)
            else:
                return None
        return visited

    def route_steps_have_outgoing_at_all_nodes(
        self,
        start_id: int | float,
        steps: list[dict[str, Any]],
    ) -> bool:

        nodes = self.intersection_ids_on_route_steps(start_id, steps)
        if nodes is None:
            return False
        return all(self.has_outgoing_edges(node_id) for node_id in nodes)

    def shortest_path(self, start_id: int | float, end_id: int | float) -> GraphPath | None:
        return self._shortest_path(start_id, end_id, {}, set())

    def shortest_path_route(
        self,
        start_id: int | float,
        end_id: int | float,
        vehicle_properties: dict[str, Any],
    ) -> RouteCandidate | None:
        path = self._shortest_path(start_id, end_id, vehicle_properties, set())
        if path is None:
            return None
        return self.route_metrics(path, vehicle_properties)

    def travel_times_from(
        self,
        start_id: int | float,
        vehicle_properties: dict[str, Any],
    ) -> dict[int, float]:
        start = int(start_id)
        if start not in self.intersections:
            return {}

        counter = itertools.count()
        queue: list[tuple[float, int, int]] = [(0.0, next(counter), start)]
        best_distances: dict[int, float] = {start: 0.0}

        while queue:
            distance, _, node_id = heapq.heappop(queue)
            if distance > best_distances.get(node_id, float("inf")):
                continue

            for next_id, road in self._adjacency.get(node_id, []):
                next_distance = distance + road.duration_for_vehicle(vehicle_properties)
                if next_distance < best_distances.get(next_id, float("inf")):
                    best_distances[next_id] = next_distance
                    heapq.heappush(queue, (next_distance, next(counter), next_id))

        return best_distances

    def route_candidates(
        self,
        start_id: int | float,
        end_id: int | float,
        vehicle_properties: dict[str, Any],
        max_candidates: int,
        search_limit: int,
    ) -> list[RouteCandidate]:
        cache_key = (
            int(start_id),
            int(end_id),
            routing_properties_key(vehicle_properties),
            max(1, max_candidates),
            max(max(1, max_candidates), search_limit),
            self._routing_revision,
        )
        cached = self._route_candidate_cache.get(cache_key)
        if cached is not None:
            return cached

        path_count = max(1, max_candidates)
        limit = max(path_count, search_limit)
        if path_count == 1 and limit == 1:
            base_path = self._shortest_path(start_id, end_id, vehicle_properties, set())
            if base_path is None:
                self._route_candidate_cache[cache_key] = []
                return []
            candidates = [self.route_metrics(base_path, vehicle_properties)]
            self._route_candidate_cache[cache_key] = candidates
            return candidates

        paths: list[GraphPath] = []
        seen: set[tuple[int, ...]] = set()

        base_path = self._shortest_path(start_id, end_id, vehicle_properties, set())
        if base_path is None:
            return []

        self._append_unique_path(paths, seen, base_path, start_id)
        queue: list[tuple[float, int, GraphPath]] = []
        counter = itertools.count()
        tried_bans: set[int] = set()

        for road in base_path.edge_list:
            if len(tried_bans) >= limit:
                break
            tried_bans.add(road.id)
            alternate = self._shortest_path(start_id, end_id, vehicle_properties, {road.id})
            if alternate is not None and self._path_key(alternate) not in seen:
                heapq.heappush(queue, (alternate.weight, next(counter), alternate))

        while queue and len(paths) < path_count:
            _, _, path = heapq.heappop(queue)
            if not self._append_unique_path(paths, seen, path, start_id):
                continue

            if len(tried_bans) >= limit:
                continue

            for road in path.edge_list:
                if road.id in tried_bans:
                    continue
                tried_bans.add(road.id)
                alternate = self._shortest_path(start_id, end_id, vehicle_properties, {road.id})
                if alternate is not None and self._path_key(alternate) not in seen:
                    heapq.heappush(queue, (alternate.weight, next(counter), alternate))
                if len(tried_bans) >= limit:
                    break

        candidates = [
            self.route_metrics(path, vehicle_properties) for path in paths[:path_count]
        ]
        self._route_candidate_cache[cache_key] = candidates
        return candidates

    def route_metrics(
        self,
        path: GraphPath,
        vehicle_properties: dict[str, Any],
    ) -> RouteCandidate:
        distance = sum(road.length for road in path.edge_list)
        travel_time = sum(road.duration_for_vehicle(vehicle_properties) for road in path.edge_list)
        co2_emissions = sum(road.emissions_for_vehicle(vehicle_properties) for road in path.edge_list)
        distance_cost = sum(road.distance_cost_for_vehicle(vehicle_properties) for road in path.edge_list)
        return RouteCandidate(
            path=GraphPath(path.edge_list, travel_time),
            distance=distance,
            travel_time=travel_time,
            co2_emissions=co2_emissions,
            distance_cost=distance_cost,
        )

    def _shortest_path(
        self,
        start_id: int | float,
        end_id: int | float,
        vehicle_properties: dict[str, Any],
        banned_road_ids: set[int],
    ) -> GraphPath | None:
        start = int(start_id)
        end = int(end_id)
        if start not in self.intersections or end not in self.intersections:
            return None
        if start == end:
            if not self.has_outgoing_edges(start):
                return None
            return GraphPath(edge_list=[], weight=0.0)

        counter = itertools.count()
        queue: list[tuple[float, int, int]] = [(0.0, next(counter), start)]
        best_distances: dict[int, float] = {start: 0.0}
        predecessor: dict[int, tuple[int, Road] | None] = {start: None}

        while queue:
            distance, _, node_id = heapq.heappop(queue)
            if node_id == end:
                path = self._graph_path_from_predecessors(predecessor, end, distance)
                if not self.graph_path_has_outgoing_at_all_nodes(start, path):
                    return None
                return path
            if distance > best_distances.get(node_id, float("inf")):
                continue

            for next_id, road in self._adjacency.get(node_id, []):
                if road.id in banned_road_ids:
                    continue
                next_distance = distance + road.duration_for_vehicle(vehicle_properties)
                if next_distance < best_distances.get(next_id, float("inf")):
                    best_distances[next_id] = next_distance
                    predecessor[next_id] = (node_id, road)
                    heapq.heappush(queue, (next_distance, next(counter), next_id))

        return None

    @staticmethod
    def _graph_path_from_predecessors(
        predecessor: dict[int, tuple[int, Road] | None],
        end: int,
        weight: float,
    ) -> GraphPath:
        edge_list: list[Road] = []
        node_id = end
        while predecessor[node_id] is not None:
            previous_id, road = predecessor[node_id]
            edge_list.append(road)
            node_id = previous_id
        edge_list.reverse()
        return GraphPath(edge_list=edge_list, weight=weight)

    def _invalidate_routing_cache(self) -> None:
        self._routing_revision += 1
        self._route_candidate_cache.clear()

    @staticmethod
    def _path_key(path: GraphPath) -> tuple[int, ...]:
        return tuple(road.id for road in path.edge_list)

    def _append_unique_path(
        self,
        paths: list[GraphPath],
        seen: set[tuple[int, ...]],
        path: GraphPath,
        start_id: int | float,
    ) -> bool:
        key = self._path_key(path)
        if key in seen:
            return False
        if not self.graph_path_has_outgoing_at_all_nodes(start_id, path):
            return False
        seen.add(key)
        paths.append(path)
        return True

    def _road_exit_node(self, entry_node: int, road: Road) -> int | None:
        for next_id, candidate in self._adjacency.get(entry_node, []):
            if candidate.id == road.id:
                return next_id
        return None
