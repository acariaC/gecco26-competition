from __future__ import annotations

import logging
import os
from pathlib import Path

from .config import EvolutionarySettings
from .dispatcher import EventDispatcher
from .events import Event
from .metrics_logger import MetricsLogger
from .optimizer import MasterEvolutionaryLoop
from .stomp_client import StompClient
from .world import WorldManager

DEFAULT_STOMP_SERVER_WS = "ws://localhost:8088/simulation-websocket"
DEFAULT_STOMP_SERVER_HTTP = "http://localhost:8088"
DEFAULT_RECONNECT_DELAY_SECONDS = 5.0


def _int_env(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _float_env(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    websocket_url = os.getenv("STOMP_SERVER_WS", DEFAULT_STOMP_SERVER_WS)
    http_url = os.getenv("STOMP_SERVER_HTTP", DEFAULT_STOMP_SERVER_HTTP)
    reconnect_delay_seconds = _float_env(
        "RECONNECT_DELAY_SECONDS",
        DEFAULT_RECONNECT_DELAY_SECONDS,
    )
    default_settings = EvolutionarySettings()
    settings = EvolutionarySettings(
        population_size=_int_env("EA_POPULATION_SIZE", default_settings.population_size),
        generations=_int_env("EA_GENERATIONS", default_settings.generations),
        seed=_int_env("EA_SEED", default_settings.seed),
        slave_count=_int_env("EA_SLAVE_COUNT", default_settings.slave_count),
        unserved_customer_penalty=_float_env(
            "EA_UNSERVED_CUSTOMER_PENALTY",
            default_settings.unserved_customer_penalty,
        ),
        route_candidates_per_leg=_int_env(
            "EA_ROUTE_CANDIDATES_PER_LEG",
            default_settings.route_candidates_per_leg,
        ),
        route_search_limit=_int_env("EA_ROUTE_SEARCH_LIMIT", default_settings.route_search_limit),
        time_weight=_float_env("EA_TIME_WEIGHT", default_settings.time_weight),
        cost_weight=_float_env("EA_COST_WEIGHT", default_settings.cost_weight),
        co2_weight=_float_env("EA_CO2_WEIGHT", default_settings.co2_weight),
        throughput_weight=_float_env("EA_THROUGHPUT_WEIGHT", default_settings.throughput_weight),
        late_service_penalty=_float_env(
            "EA_LATE_SERVICE_PENALTY",
            default_settings.late_service_penalty,
        ),
        warm_start_cache_size=_int_env(
            "EA_WARM_START_CACHE_SIZE",
            default_settings.warm_start_cache_size,
        ),
        continuous_reoptimization_interval=_int_env(
            "EA_CONTINUOUS_REOPTIMIZATION_INTERVAL",
            default_settings.continuous_reoptimization_interval,
        ),
        trivial_search_threshold=_int_env(
            "EA_TRIVIAL_SEARCH_THRESHOLD",
            default_settings.trivial_search_threshold,
        ),
        replan_debounce_seconds=_int_env(
            "EA_REPLAN_DEBOUNCE_SECONDS",
            default_settings.replan_debounce_seconds,
        ),
        eval_workers=_int_env("EA_EVAL_WORKERS", default_settings.eval_workers),
        max_candidate_vehicles_per_request=_int_env(
            "EA_MAX_CANDIDATE_VEHICLES_PER_REQUEST",
            default_settings.max_candidate_vehicles_per_request,
        ),
        enable_idle_repositioning=_bool_env(
            "EA_ENABLE_IDLE_REPOSITIONING",
            default_settings.enable_idle_repositioning,
        ),
        idle_min_spacing_meters=_float_env(
            "EA_IDLE_MIN_SPACING_METERS",
            default_settings.idle_min_spacing_meters,
        ),
        max_repositions_per_event=_int_env(
            "EA_MAX_REPOSITIONS_PER_EVENT",
            default_settings.max_repositions_per_event,
        ),
        idle_reposition_threshold_seconds=_int_env(
            "EA_IDLE_REPOSITION_THRESHOLD_SECONDS",
            default_settings.idle_reposition_threshold_seconds,
        ),
    ).normalized()

    world = WorldManager(http_url)
    distributed = _bool_env("MAPE_K_DISTRIBUTED", False)
    metrics_output_dir = Path(os.getenv("METRICS_OUTPUT_DIR", str(Path.cwd() / "metrics")))

    with MasterEvolutionaryLoop(world, settings, distributed=distributed) as optimizer:
        metrics_logger = MetricsLogger(output_dir=metrics_output_dir)
        dispatcher = EventDispatcher(world, optimizer, metrics_logger=metrics_logger)
        def on_connected() -> list[Event]:
            if world.ensure_road_network_loaded():
                return [Event("client", "initialized")]
            return []

        client = StompClient(
            websocket_url,
            dispatcher,
            reconnect_delay_seconds,
            on_connected=on_connected,
        )

        try:
            client.run_forever()
        except KeyboardInterrupt:
            logging.getLogger(__name__).info("Stopping evolutionary optimizer.")
            client.stop()
            metrics_logger.flush()


if __name__ == "__main__":
    main()
