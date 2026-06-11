from __future__ import annotations

import json
from typing import Callable
from wsgiref.simple_server import make_server

from .explainability import ExplanationRecord, STORE, to_jsonable


def _read_body(environ: dict) -> bytes:
    try:
        length = int(environ.get("CONTENT_LENGTH") or 0)
    except (ValueError, TypeError):
        length = 0
    if length > 0:
        return environ["wsgi.input"].read(length)
    return b""


def application(environ: dict, start_response: Callable):
    path = environ.get("PATH_INFO", "")
    method = environ.get("REQUEST_METHOD", "GET")
    cors_headers = [("Access-Control-Allow-Origin", "*"), ("Access-Control-Allow-Methods", "GET,POST,OPTIONS"), ("Access-Control-Allow-Headers", "Content-Type")]

                           
    if method == "OPTIONS":
        start_response("204 No Content", cors_headers)
        return [b""]

    if path.startswith("/api/explanations"):
                                
        if method == "POST" and path == "/api/explanations":
            body = _read_body(environ)
            try:
                payload = json.loads(body.decode("utf-8"))
            except Exception:
                start_response("400 Bad Request", [("Content-Type", "application/json")] + cors_headers)
                return [b"{\"error\":\"invalid json\"}"]
            try:
                record = ExplanationRecord(**payload)
            except TypeError:
                start_response("400 Bad Request", [("Content-Type", "application/json")] + cors_headers)
                return [b"{\"error\":\"invalid explanation record\"}"]
            STORE.add(record)
            start_response("201 Created", [("Content-Type", "application/json")] + cors_headers)
            return [json.dumps({"status": "ok"}).encode("utf-8")]

                             
        if method == "GET":
                               
            if path == "/api/explanations":
                data = [to_jsonable(r) for r in STORE.list_all()]
                start_response("200 OK", [("Content-Type", "application/json")] + cors_headers)
                return [json.dumps(data).encode("utf-8")]

                                             
            parts = path.strip("/").split("/")
            if len(parts) == 4 and parts[2] == "requests":
                request_id = parts[3]
                data = [to_jsonable(r) for r in STORE.by_request(request_id)]
                start_response("200 OK", [("Content-Type", "application/json")] + cors_headers)
                return [json.dumps(data).encode("utf-8")]

            if len(parts) == 4 and parts[2] == "vehicles":
                vehicle_id = parts[3]
                data = [to_jsonable(r) for r in STORE.by_vehicle(vehicle_id)]
                start_response("200 OK", [("Content-Type", "application/json")] + cors_headers)
                return [json.dumps(data).encode("utf-8")]

    start_response("404 Not Found", [("Content-Type", "application/json")] + cors_headers)
    return [b"{\"error\":\"not found\"}"]


def run_server(host: str = "127.0.0.1", port: int = 8001):
    print(f"Starting explanations HTTP server on http://{host}:{port}")
    with make_server(host, port, application) as httpd:
        httpd.serve_forever()


if __name__ == "__main__":
    run_server()
