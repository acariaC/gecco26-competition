from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Event:


    category: str
    name: str
    data: dict[str, Any] = field(default_factory=dict)
    explanations: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "Event":
        return cls(
            category=str(payload["category"]),
            name=str(payload["name"]),
            data=dict(payload.get("data") or {}),
            explanations=dict(payload.get("explanations") or {}),
        )

    def to_mapping(self) -> dict[str, Any]:
        mapping: dict[str, Any] = {
            "category": self.category,
            "name": self.name,
            "data": self.data,
        }
        if self.explanations:
            mapping["explanations"] = self.explanations
        return mapping

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def put(self, key: str, value: Any) -> None:
        self.data[key] = value
