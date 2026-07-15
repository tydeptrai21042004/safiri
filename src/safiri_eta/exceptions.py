from __future__ import annotations

from typing import Any


def detect_exceptions(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    exceptions: list[dict[str, Any]] = []

    def add(code: str, severity: str, evidence: str) -> None:
        exceptions.append({"type": code, "severity": severity, "evidence": evidence})

    missing = int(snapshot.get("missing_event_count", 0))
    event_age = float(snapshot.get("event_age_hours", 0.0))
    late = int(snapshot.get("late_event_count", 0))
    reliability = float(snapshot.get("source_reliability", 1.0))
    congestion = float(snapshot.get("congestion_level", 0.0))
    documents_ready = int(snapshot.get("documents_ready", 1))
    current_delay = float(snapshot.get("current_delay_hours", 0.0))

    if missing > 0:
        add("MISSING_MILESTONE", "MEDIUM" if missing == 1 else "HIGH", f"{missing} expected milestone(s) are unavailable")
    if event_age > 6 or late > 0:
        add("LATE_REPORTING", "MEDIUM", f"Latest event age is {event_age:.1f} h; {late} late report(s) observed")
    if reliability < 0.78:
        add("LOW_SOURCE_RELIABILITY", "MEDIUM", f"Mean source reliability is {reliability:.2f}")
    if congestion >= 0.80:
        add("HIGH_CONGESTION", "HIGH", f"Congestion level is {congestion:.2f}")
    if not documents_ready and snapshot.get("current_stage") in {"DISCHARGED", "CUSTOMS_STARTED"}:
        add("DOCUMENTS_NOT_READY", "HIGH", "Customs documents are not ready")
    if current_delay > 18:
        add("PROPAGATED_DELAY", "HIGH", f"Current milestone is already {current_delay:.1f} h late")
    return exceptions


def recommended_actions(snapshot: dict[str, Any], delay_probability: float) -> list[str]:
    actions: list[str] = []
    if int(snapshot.get("documents_ready", 1)) == 0:
        actions.append("Confirm customs-document readiness and missing declarations.")
    if float(snapshot.get("congestion_level", 0.0)) >= 0.70:
        actions.append("Check terminal capacity and reserve an alternative pickup slot.")
    if int(snapshot.get("missing_event_count", 0)) > 0 or float(snapshot.get("event_age_hours", 0.0)) > 6:
        actions.append("Request a fresh status update from the responsible data source.")
    if delay_probability >= 0.60:
        actions.append("Escalate the shipment to the exception-management queue.")
    if not actions:
        actions.append("Continue monitoring; no immediate intervention is recommended.")
    return actions


def data_quality(snapshot: dict[str, Any]) -> dict[str, Any]:
    penalties = 0.0
    issues: list[str] = []
    missing = int(snapshot.get("missing_event_count", 0))
    late = int(snapshot.get("late_event_count", 0))
    reliability = float(snapshot.get("source_reliability", 1.0))
    if missing:
        penalties += min(0.35, missing * 0.12)
        issues.append(f"{missing} expected milestone(s) missing")
    if late:
        penalties += min(0.25, late * 0.08)
        issues.append(f"{late} event(s) reported more than 6 hours late")
    if reliability < 0.85:
        penalties += min(0.25, (0.85 - reliability) * 1.5)
        issues.append(f"Mean source reliability is {reliability:.2f}")
    score = max(0.0, min(1.0, 1.0 - penalties))
    level = "HIGH" if score >= 0.82 else "MEDIUM" if score >= 0.60 else "LOW"
    return {"score": round(score, 3), "level": level, "issues": issues}

