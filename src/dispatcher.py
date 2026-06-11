from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Protocol

from .events import Event
from .metrics_logger import MetricsLogger
from .world import WorldManager

LOGGER = logging.getLogger(__name__)
EventHandler = Callable[[Event], list[Event]]


class OptimizerHandlers(Protocol):
    def plan_route(self, event: Event) -> list[Event]: ...

    def add_vehicle(self, event: Event) -> list[Event]: ...

    def vehicle_available(self, event: Event) -> list[Event]: ...

    def update_vehicle_position(self, event: Event) -> list[Event]: ...

    def update_time(self, event: Event) -> list[Event]: ...

    def road_network_changed(self, event: Event) -> list[Event]: ...

    def on_reset(self, event: Event) -> list[Event]: ...

    def remove_request(self, event: Event) -> list[Event]: ...

    def record_person(self, event: Event) -> list[Event]: ...

    def record_pickup(self, event: Event) -> list[Event]: ...

    def cancel_request(self, event: Event) -> list[Event]: ...


class EventDispatcher:


    def __init__(
        self,
        world: WorldManager,
        optimizer: OptimizerHandlers,
        metrics_logger: MetricsLogger | None = None,
    ) -> None:
        self._world = world
        self._optimizer = optimizer
        self._metrics_logger = metrics_logger or MetricsLogger()
        self._event_handlers: dict[tuple[str, str], EventHandler] = {
            ("road-network", "changed-road-property"): self._change_road_properties,
            ("road-network", "changed-intersection-property"): self._change_intersection_properties,
            ("metrics", "metrics"): self._record_metrics,
            ("person", "added"): optimizer.record_person,
            ("person", "error"): self._log_error,
            ("person", "removed"): self._noop,
            ("request", "ride-request-received"): optimizer.plan_route,
            ("request", "ride-request-served"): optimizer.remove_request,
            ("request", "customer-not-served"): optimizer.cancel_request,
            ("request", "ride-request-expired"): optimizer.cancel_request,
            ("simulation", "reset"): self._reset_simulation,
            ("simulation", "initialize"): world.fetch_road_network,
            ("simulation", "state"): self._noop,
            ("simulation", "stop"): self._stop_simulation,
            ("taxi-fleet", "added-taxi"): optimizer.add_vehicle,
            ("taxi-fleet", "error"): self._log_error,
            ("taxi-fleet", "plan-route"): self._noop,
            ("taxi-fleet", "picked-up-passengers"): optimizer.record_pickup,
            ("taxi-fleet", "dropped-off-passengers"): self._noop,
            ("time", "time"): optimizer.update_time,
            ("vehicle", "added"): self._noop,
            ("vehicle", "error"): self._log_error,
            ("vehicle", "finished-move"): optimizer.vehicle_available,
            ("vehicle", "move"): self._noop,
            ("vehicle", "passed-intersection"): optimizer.update_vehicle_position,
            ("vehicle", "route-event"): self._noop,
            ("vehicle", "route-planned"): self._noop,
            ("request", "customer-arrived"): self._noop,
        }

    def handle_event(self, event: Event) -> list[Event]:
        handler = self._event_handlers.get((event.category, event.name))
        if handler is None:
            LOGGER.warning("No handler for event %s found.", event)
            return []

        return handler(event)

    def _change_road_properties(self, event: Event) -> list[Event]:
        self._world.change_road_properties(event)
        return self._optimizer.road_network_changed(event)

    def _change_intersection_properties(self, event: Event) -> list[Event]:
        self._world.change_intersection_properties(event)
        return self._optimizer.road_network_changed(event)

    @staticmethod
    def _noop(event: Event) -> list[Event]:
        return []

    @staticmethod
    def _log_error(event: Event) -> list[Event]:
        reason = event.data.get("reason", "unknown")
        LOGGER.warning(
            "Simulator %s:%s error: reason=%s data=%s",
            event.category,
            event.name,
            reason,
            event.data,
        )
        return []

    def _record_metrics(self, event: Event) -> list[Event]:
        self._metrics_logger.record(event)
        return []

    def _reset_simulation(self, event: Event) -> list[Event]:
        self._metrics_logger.reset()
        return self._optimizer.on_reset(event)

    def _stop_simulation(self, event: Event) -> list[Event]:
        self._metrics_logger.flush()
        return []
