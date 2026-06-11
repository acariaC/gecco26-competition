from __future__ import annotations

import logging
from dataclasses import dataclass

from .events import Event
from .world import WorldManager

LOGGER = logging.getLogger(__name__)


@dataclass
class ValidationResult:


    valid: bool
    reason: str | None = None
    reject_event: Event | None = None


class RequestValidator:


    def __init__(self, world: WorldManager) -> None:
        self._world = world
        self._dead_ends: set[int] = set()

    @property
    def dead_end_count(self) -> int:

        return len(self._dead_ends)

    def refresh_dead_ends(self) -> None:

        self._dead_ends = self._world.dead_end_intersection_ids()
        LOGGER.info("Graph analysis: %d dead-end intersections found.", len(self._dead_ends))

    def validate_request(
        self,
        request_id: str,
        pickup_id: int,
        dropoff_id: int,
    ) -> ValidationResult:


        if pickup_id in self._dead_ends:
            reason = f"pickup intersection {pickup_id} is a dead-end (no outgoing roads)"
            LOGGER.warning("Rejecting request %s: %s", request_id, reason)
            return ValidationResult(
                valid=False,
                reason=reason,
                reject_event=self._create_reject_event(request_id, reason),
            )

        if dropoff_id in self._dead_ends:
            reason = f"dropoff intersection {dropoff_id} is a dead-end (no outgoing roads)"
            LOGGER.warning("Rejecting request %s: %s", request_id, reason)
            return ValidationResult(
                valid=False,
                reason=reason,
                reject_event=self._create_reject_event(request_id, reason),
            )

        return ValidationResult(valid=True)

    @staticmethod
    def _create_reject_event(request_id: str, reason: str) -> Event:

        event = Event("request", "reject")
        event.put("id", request_id)
        event.put("explanation", {"reason": reason})
        return event
