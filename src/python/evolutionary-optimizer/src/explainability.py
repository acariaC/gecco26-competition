from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List


@dataclass
class CandidateEvaluation:
    candidate_id: str
    vehicle_id: str
    raw_metrics: Dict[str, float]
    normalized_metrics: Dict[str, float]
    contributions: Dict[str, float]
    score: float
    rank: int = 0
    compared_to_next: Dict[str, float] = field(default_factory=dict)


@dataclass
class ExplanationRecord:
    id: str
    request_id: str
    vehicle_id: str
    timestamp: str
    algorithm: str
    final_score: float
    textual_explanation: str
    details: Dict[str, Any]


class ExplanationStore:
    def __init__(self, max_size: int = 500) -> None:
        self._lock = threading.Lock()
        self._store: List[ExplanationRecord] = []
        self.max_size = max_size

    def add(self, record: ExplanationRecord) -> None:
        with self._lock:
            self._store.append(record)
            if len(self._store) > self.max_size:
                             
                self._store = self._store[-self.max_size :]

    def list_all(self) -> List[ExplanationRecord]:
        with self._lock:
            return list(self._store)

    def by_request(self, request_id: str) -> List[ExplanationRecord]:
        with self._lock:
            return [r for r in reversed(self._store) if r.request_id == request_id]

    def by_vehicle(self, vehicle_id: str) -> List[ExplanationRecord]:
        with self._lock:
            return [r for r in reversed(self._store) if r.vehicle_id == vehicle_id]


                                                           
STORE = ExplanationStore()


def build_textual_explanation(best: CandidateEvaluation) -> str:
                                                       
    delta_pickup = int(best.compared_to_next.get("delta_pickup_delay", 0))
    delta_travel = int(best.compared_to_next.get("delta_travel_time", 0))
    if delta_pickup < 0:
        pickup_phrase = f"pickup delay {abs(delta_pickup)}s lower than next-best"
    elif delta_pickup > 0:
        pickup_phrase = f"pickup delay {delta_pickup}s higher than next-best"
    else:
        pickup_phrase = "pickup delay similar to next-best"

    travel_phrase = f"travel time {delta_travel:+d}s compared to next-best"

    return f"Selected {best.vehicle_id} because {pickup_phrase}, {travel_phrase}."


def make_explanation_record(request_id: str, best: CandidateEvaluation, details: Dict[str, Any]) -> ExplanationRecord:
    rec = ExplanationRecord(
        id=str(uuid.uuid4()),
        request_id=request_id,
        vehicle_id=best.vehicle_id,
        timestamp=datetime.utcnow().isoformat() + "Z",
        algorithm="ea-v1",
        final_score=float(best.score),
        textual_explanation=build_textual_explanation(best),
        details=details,
    )
    return rec


def to_jsonable(record: ExplanationRecord) -> Dict[str, Any]:
                                                 
    result = asdict(record)
    return result
