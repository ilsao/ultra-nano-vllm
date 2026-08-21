from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SchedulerPolicyMetrics:
    """Counters maintained by a scheduler implementation."""

    candidate_inspections: int = 0
    backfill_selections: int = 0
    preemptions: int = 0
    preempted_blocks: int = 0


@dataclass(frozen=True, slots=True)
class SchedulerMetrics:
    """Per-generation scheduler and batch-shape telemetry."""

    scheduler_time: float = 0.0
    schedule_calls: int = 0
    prefill_steps: int = 0
    decode_steps: int = 0
    prefill_scheduled_tokens: int = 0
    prefill_capacity_tokens: int = 0
    prefill_scheduled_sequences: int = 0
    extra_prefill_tokens: int = 0
    candidate_inspections: int = 0
    backfill_selections: int = 0
    preemptions: int = 0
    preempted_blocks: int = 0
