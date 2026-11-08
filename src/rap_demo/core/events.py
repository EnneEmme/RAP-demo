"""EventBus and lifecycle event hooks for MCTS planning iterations."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Callable, Optional


class EventType(str, Enum):
    ON_SELECT = "on_select"
    ON_EXPAND = "on_expand"
    ON_SIMULATE = "on_simulate"
    ON_BACKPROP = "on_backprop"
    ON_ITERATION_COMPLETE = "on_iteration_complete"


@dataclass
class Event:
    type: EventType
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class EventBus:
    """Publish-subscribe event bus for tracing MCTS lifecycle events."""

    def __init__(self) -> None:
        self._listeners: dict[EventType, list[Callable[[Event], None]]] = {}

    def subscribe(self, event_type: EventType | str, callback: Callable[[Event], None]) -> None:
        """Register a callback for a specific event type."""
        ev_type = EventType(event_type) if isinstance(event_type, str) else event_type
        if ev_type not in self._listeners:
            self._listeners[ev_type] = []
        self._listeners[ev_type].append(callback)

    def unsubscribe(self, event_type: EventType | str, callback: Callable[[Event], None]) -> None:
        """Remove a previously registered callback."""
        ev_type = EventType(event_type) if isinstance(event_type, str) else event_type
        if ev_type in self._listeners:
            try:
                self._listeners[ev_type].remove(callback)
            except ValueError:
                pass

    def emit(self, event_type: EventType | str, data: Optional[dict[str, Any]] = None) -> None:
        """Emit an event to all registered listeners."""
        ev_type = EventType(event_type) if isinstance(event_type, str) else event_type
        event = Event(type=ev_type, data=data if data is not None else {})
        if ev_type in self._listeners:
            for callback in list(self._listeners[ev_type]):
                callback(event)

    def clear(self) -> None:
        """Remove all event listeners."""
        self._listeners.clear()
