from __future__ import annotations

MILESTONES = [
    "BOOKED",
    "ORIGIN_DEPARTED",
    "PORT_ARRIVED",
    "DISCHARGED",
    "CUSTOMS_STARTED",
    "CUSTOMS_CLEARED",
    "GATE_OUT",
    "DELIVERED",
]

MILESTONE_INDEX = {name: index for index, name in enumerate(MILESTONES)}

STATUS_VARIANTS = {
    "BOOKED": ["Booking confirmed", "BOOKED", "Shipment booked"],
    "ORIGIN_DEPARTED": ["Origin departed", "DEPARTED_ORIGIN", "Left origin"],
    "PORT_ARRIVED": ["Port arrival", "ARRIVED_PORT", "Arrived at terminal"],
    "DISCHARGED": ["Container discharged", "DISCH", "Vessel unload"],
    "CUSTOMS_STARTED": ["Customs started", "CUSTOMS_IN", "Declaration lodged"],
    "CUSTOMS_CLEARED": ["Customs cleared", "CUSTOMS_RELEASE", "Released by customs"],
    "GATE_OUT": ["Gate out", "GATE_OUT", "Container left terminal"],
    "DELIVERED": ["Delivered", "FINAL_DELIVERY", "Proof of delivery"],
}

STATUS_TO_CANONICAL = {
    variant.lower(): canonical
    for canonical, variants in STATUS_VARIANTS.items()
    for variant in variants
}

NUMERIC_FEATURES = [
    "event_index",
    "stage_progress",
    "hours_since_booking",
    "planned_remaining_hours",
    "current_delay_hours",
    "event_age_hours",
    "congestion_level",
    "weather_severity",
    "documents_ready",
    "observed_event_count",
    "missing_event_count",
    "late_event_count",
    "duplicate_event_count",
    "source_reliability",
    "booking_month",
    "booking_weekday",
    "booking_hour",
    "is_weekend",
]

CATEGORICAL_FEATURES = [
    "route",
    "mode",
    "cargo_type",
    "service_level",
    "current_stage",
]

STAGE_CATEGORICAL_FEATURES = CATEGORICAL_FEATURES + ["target_stage"]
STAGE_NUMERIC_FEATURES = NUMERIC_FEATURES + ["target_stage_index", "is_first_remaining_stage"]

