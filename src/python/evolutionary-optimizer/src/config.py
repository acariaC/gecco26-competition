from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvolutionarySettings:


    population_size: int = 24
    generations: int = 12
    seed: int = 7
    slave_count: int = 1
    unserved_customer_penalty: float = 1_000_000_000.0
    route_candidates_per_leg: int = 2
    route_search_limit: int = 12
    time_weight: float = 0.4
    cost_weight: float = 0.0
    co2_weight: float = 0.4
    throughput_weight: float = 0.2
    late_service_penalty: float = 10_000.0
    warm_start_cache_size: int = 40
    continuous_reoptimization_interval: int = 20
    trivial_search_threshold: int = 64
    replan_debounce_seconds: int = 2
    eval_workers: int = 1
    max_candidate_vehicles_per_request: int = 4
    enable_idle_repositioning: bool = True
    idle_min_spacing_meters: float = 1500.0
    max_repositions_per_event: int = 3
    idle_reposition_threshold_seconds: int = 300

    def normalized(self) -> "EvolutionarySettings":
        return EvolutionarySettings(
            population_size=max(4, self.population_size),
            generations=max(1, self.generations),
            seed=self.seed,
            slave_count=max(1, self.slave_count),
            unserved_customer_penalty=max(1.0, self.unserved_customer_penalty),
            route_candidates_per_leg=max(1, self.route_candidates_per_leg),
            route_search_limit=max(1, self.route_search_limit),
            time_weight=max(0.0, self.time_weight),
            cost_weight=max(0.0, self.cost_weight),
            co2_weight=max(0.0, self.co2_weight),
            throughput_weight=max(0.0, self.throughput_weight),
            late_service_penalty=max(0.0, self.late_service_penalty),
            warm_start_cache_size=max(0, self.warm_start_cache_size),
            continuous_reoptimization_interval=max(0, self.continuous_reoptimization_interval),
            trivial_search_threshold=max(0, self.trivial_search_threshold),
            replan_debounce_seconds=max(0, self.replan_debounce_seconds),
            eval_workers=max(1, self.eval_workers),
            max_candidate_vehicles_per_request=max(0, self.max_candidate_vehicles_per_request),
            enable_idle_repositioning=self.enable_idle_repositioning,
            idle_min_spacing_meters=max(0.0, self.idle_min_spacing_meters),
            max_repositions_per_event=max(0, self.max_repositions_per_event),
            idle_reposition_threshold_seconds=max(0, self.idle_reposition_threshold_seconds),
        )
