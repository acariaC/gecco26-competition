from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

from websocket import WebSocketApp

from .dispatcher import EventDispatcher
from .events import Event

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class StompFrame:
    command: str
    headers: dict[str, str]
    body: str


class StompClient:


    def __init__(
        self,
        url: str,
        event_dispatcher: EventDispatcher,
        reconnect_delay_seconds: float = 5.0,
        on_connected: Callable[[], list[Event]] | None = None,
    ) -> None:
        self.url = url
        self.event_dispatcher = event_dispatcher
        self.reconnect_delay_seconds = reconnect_delay_seconds
        self.on_connected = on_connected
        self._app: WebSocketApp | None = None
        self._send_lock = threading.Lock()
        self._stopped = threading.Event()

    def run_forever(self) -> None:
        while not self._stopped.is_set():
            LOGGER.info("Connecting to %s", self.url)
            self._app = WebSocketApp(
                self.url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
            )
            self._app.run_forever()

            if not self._stopped.is_set():
                LOGGER.info("Reconnecting in %s seconds.", self.reconnect_delay_seconds)
                time.sleep(self.reconnect_delay_seconds)

    def stop(self) -> None:
        self._stopped.set()
        if self._app is not None:
            self._app.close()

    def send_event(self, event: Event) -> None:
        body = json.dumps(event.to_mapping(), separators=(",", ":"))
        headers = {
            "destination": "/sim/events",
            "category": event.category,
            "name": event.name,
            "content-type": "application/json",
        }
        self._send_frame("SEND", headers, body)

    def _on_open(self, app: WebSocketApp) -> None:
        parsed_url = urlparse(self.url)
        headers = {
            "accept-version": "1.2",
            "host": parsed_url.hostname or "localhost",
            "heart-beat": "0,0",
            "role": "OPTIMIZER",
        }
        self._send_frame("CONNECT", headers)

    def _on_message(self, app: WebSocketApp, message: str | bytes) -> None:
        if isinstance(message, bytes):
            message = message.decode("utf-8")
        if message.strip() == "":
            return

        for frame in _parse_frames(message):
            if frame.command == "CONNECTED":
                LOGGER.info("Connected to STOMP server.")
                self._send_frame(
                    "SUBSCRIBE",
                    {
                        "id": "simulation-events",
                        "destination": "/topic/simulation-events",
                        "ack": "auto",
                    },
                )
                if self.on_connected is not None:
                    for response in self.on_connected():
                        LOGGER.info("Sending %s.", response)
                        self.send_event(response)
                continue

            if frame.command == "MESSAGE":
                self._handle_message(frame)
                continue

            if frame.command == "ERROR":
                LOGGER.error("STOMP error: headers=%s body=%s", frame.headers, frame.body)

    def _handle_message(self, frame: StompFrame) -> None:
        try:
            event = Event.from_mapping(json.loads(frame.body))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            LOGGER.exception("Failed to decode simulator event: %s", error)
            return

        LOGGER.debug(
            "Received event %s:%s from topic %s.",
            event.category,
            event.name,
            frame.headers.get("destination"),
        )

        responses = self.event_dispatcher.handle_event(event)
        for response in responses:
            LOGGER.info("Sending %s.", response)
            self.send_event(response)

    def _on_error(self, app: WebSocketApp, error: Any) -> None:
        LOGGER.error("WebSocket error: %s", error)

    def _on_close(
        self,
        app: WebSocketApp,
        close_status_code: int | None,
        close_msg: str | None,
    ) -> None:
        LOGGER.info("WebSocket closed: code=%s message=%s", close_status_code, close_msg)

    def _send_frame(
        self,
        command: str,
        headers: dict[str, str],
        body: str = "",
    ) -> None:
        app = self._app
        if app is None or app.sock is None or not app.sock.connected:
            LOGGER.warning("Cannot send STOMP frame %s, not connected.", command)
            return

        frame = _encode_frame(command, headers, body)
        with self._send_lock:
            app.send(frame)


def _encode_frame(command: str, headers: dict[str, str], body: str = "") -> str:
    header_lines = [command, *[f"{key}:{value}" for key, value in headers.items()]]
    return "\n".join(header_lines) + "\n\n" + body + "\x00"


def _parse_frames(message: str) -> list[StompFrame]:
    frames: list[StompFrame] = []
    for raw_frame in message.split("\x00"):
        if raw_frame.strip() == "":
            continue

        header_text, _, body = raw_frame.partition("\n\n")
        header_lines = header_text.splitlines()
        if not header_lines:
            continue

        command = header_lines[0].strip()
        headers: dict[str, str] = {}
        for line in header_lines[1:]:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            headers[key] = value

        frames.append(StompFrame(command, headers, body))

    return frames
