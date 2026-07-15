from __future__ import annotations

from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .config import project_path
from .utils import read_json


BLUE = colors.HexColor("#173F5F")
LIGHT_BLUE = colors.HexColor("#EAF2F8")
GRAY = colors.HexColor("#F3F4F6")
MUTED = colors.HexColor("#5B6573")


def _fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def _footer(canvas, document) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D7E0E8"))
    canvas.line(0.75 * inch, 0.52 * inch, 7.75 * inch, 0.52 * inch)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(0.75 * inch, 0.34 * inch, "SAFiRi AI take-home - reproducible synthetic-data study")
    canvas.drawRightString(7.75 * inch, 0.34 * inch, f"Page {document.page}")
    canvas.restoreState()


def generate_technical_report(config: dict[str, Any]) -> Path:
    reports_dir = project_path(config, "reports_dir")
    metrics = read_json(reports_dir / "metrics.json")
    output = reports_dir / "technical_report.pdf"
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=23,
            textColor=BLUE,
            alignment=TA_CENTER,
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Section",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=14,
            textColor=BLUE,
            spaceBefore=8,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyCompact",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=12.2,
            alignment=TA_LEFT,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Small",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=7.8,
            leading=9.8,
            textColor=MUTED,
        )
    )
    doc = SimpleDocTemplate(
        str(output),
        pagesize=letter,
        rightMargin=0.68 * inch,
        leftMargin=0.68 * inch,
        topMargin=0.62 * inch,
        bottomMargin=0.70 * inch,
        title="Milestone-Aware Shipment ETA and Delay Propagation",
        author="SAFiRi AI Intern Candidate",
    )
    story = []
    story.append(Spacer(1, 0.12 * inch))
    story.append(Paragraph("Milestone-Aware Shipment ETA and Delay Propagation", styles["ReportTitle"]))
    story.append(
        Paragraph(
            "Three-day take-home implementation: synthetic event journeys, point-in-time modelling, delay risk, local explanations, API and dashboard.",
            styles["BodyCompact"],
        )
    )
    story.append(Spacer(1, 0.08 * inch))

    story.append(Paragraph("1. Problem and design", styles["Section"]))
    story.append(
        Paragraph(
            "The system predicts the final arrival time from the events known at a given observation time. It also estimates the probability that final delivery exceeds the planned time by more than 12 hours. The core design decision is to distinguish actual_time (when an event happened) from observed_at (when the system learned it), preventing future-event leakage.",
            styles["BodyCompact"],
        )
    )
    story.append(
        Paragraph(
            "The canonical journey contains BOOKED, ORIGIN_DEPARTED, PORT_ARRIVED, DISCHARGED, CUSTOMS_STARTED, CUSTOMS_CLEARED, GATE_OUT and DELIVERED. A new point-in-time snapshot is created whenever a report arrives. All snapshots from one shipment remain in one chronological split.",
            styles["BodyCompact"],
        )
    )

    story.append(Paragraph("2. Data and delay simulation", styles["Section"]))
    dataset = metrics["dataset"]
    data_rows = [
        ["Property", "Value"],
        ["Synthetic shipments", "300"],
        ["Routes / modes", "6 / sea, air, road, multimodal"],
        ["Test shipments / snapshots", f"{dataset['test_shipments']} / {dataset['test_snapshots']}"],
        ["Test delay prevalence", f"{dataset['delay_prevalence']:.1%}"],
    ]
    table = Table(data_rows, colWidths=[2.55 * inch, 3.95 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), BLUE),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CAD5E0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, GRAY]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.08 * inch))
    story.append(
        Paragraph(
            "Stage duration combines route schedule, congestion, weather, document readiness, congestion-document interaction, random disruptions and partial recovery. Intermediate reports may be missing, duplicated or delayed. BOOKED and DELIVERED remain observable so every journey has a usable boundary and ground truth.",
            styles["BodyCompact"],
        )
    )

    story.append(Paragraph("3. Models and explanations", styles["Section"]))
    story.append(
        Paragraph(
            "Three ETA approaches are compared: route-stage medians, direct remaining-time gradient boosting, and a stage-aware gradient-boosting model that predicts every remaining transition and sums the durations. Delay risk uses a class-weighted gradient-boosting classifier followed by probability calibration on the validation set. P10-P90 ETA bounds use validation residual quantiles.",
            styles["BodyCompact"],
        )
    )
    story.append(
        Paragraph(
            "Each prediction includes remaining-stage estimates, historical medians, data-quality score, rule-based exceptions and local perturbation drivers expressed in hours. These drivers describe model sensitivity rather than causal effects. Recommendations are produced by auditable operational rules.",
            styles["BodyCompact"],
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("4. Results", styles["Section"]))
    eta = metrics["eta"]
    result_rows = [["Model", "MAE (h)", "RMSE (h)", "P90 AE (h)", "Within +/-12 h"]]
    for label, key in (("Route-stage median", "baseline"), ("Direct ETA", "direct"), ("Stage-aware ETA", "stage_aware")):
        result = eta[key]
        result_rows.append(
            [
                label,
                _fmt(result["mae_hours"]),
                _fmt(result["rmse_hours"]),
                _fmt(result["p90_ae_hours"]),
                f"{result['within_12_hours']:.1%}",
            ]
        )
    result_table = Table(result_rows, colWidths=[2.05 * inch, 0.85 * inch, 0.9 * inch, 0.95 * inch, 1.35 * inch])
    result_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), BLUE),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.3),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CAD5E0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, GRAY]),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(result_table)
    story.append(Spacer(1, 0.09 * inch))
    direct = eta["direct"]["mae_hours"]
    stage = eta["stage_aware"]["mae_hours"]
    baseline = eta["baseline"]["mae_hours"]
    story.append(
        Paragraph(
            f"The direct model achieved the lowest MAE ({direct:.2f} h). The stage-aware model reached {stage:.2f} h, improving on the median baseline ({baseline:.2f} h) while providing a decomposable prediction. This is an accuracy-interpretability trade-off rather than evidence that the more structured model is always superior.",
            styles["BodyCompact"],
        )
    )
    comparison = reports_dir / "figures" / "eta_model_comparison.png"
    if comparison.exists():
        story.append(Image(str(comparison), width=6.35 * inch, height=3.11 * inch))
        story.append(Paragraph("Figure 1. ETA model comparison on the untouched chronological test set.", styles["Small"]))

    risk = metrics["delay_risk"]
    story.append(Paragraph("5. Delay-risk classification", styles["Section"]))
    risk_rows = [
        ["Precision", "Recall", "F1", "PR-AUC", "ROC-AUC", "Brier"],
        [
            _fmt(risk["precision"]),
            _fmt(risk["recall"]),
            _fmt(risk["f1"]),
            _fmt(risk["pr_auc"]),
            _fmt(risk["roc_auc"] or 0),
            _fmt(risk["brier_score"], 3),
        ],
    ]
    risk_table = Table(risk_rows, colWidths=[1.08 * inch] * 6)
    risk_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), BLUE),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.4),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CAD5E0")),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(risk_table)
    story.append(
        Paragraph(
            "Because delayed snapshots are a minority, PR-AUC and Brier score are reported in addition to F1. The classifier is suitable for ranking an exception queue, but the decision threshold should ultimately be chosen from SAFiRi's false-alert and missed-delay costs.",
            styles["BodyCompact"],
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("6. Robustness and operational behaviour", styles["Section"]))
    robustness = reports_dir / "figures" / "missing_event_robustness.png"
    if robustness.exists():
        story.append(Image(str(robustness), width=6.45 * inch, height=3.13 * inch))
        story.append(Paragraph("Figure 2. Sensitivity test using increasingly adverse missing-event indicators.", styles["Small"]))
    story.append(
        Paragraph(
            "The robustness test changes missing-event count and source reliability without exposing future events. It is a stress test of model sensitivity, not a substitute for re-running full event deletion and snapshot reconstruction. The repository keeps this limitation explicit in metrics.json.",
            styles["BodyCompact"],
        )
    )

    story.append(Paragraph("7. Demo and reproducibility", styles["Section"]))
    demo_rows = [
        ["Component", "Implementation"],
        ["Data pipeline", "Raw -> validated canonical events -> point-in-time snapshots"],
        ["Training", "One command: python -m safiri_eta.cli all"],
        ["API", "FastAPI: health, shipments, timeline, prediction, simulation"],
        ["Dashboard", "Streamlit operations view, shipment explanation and what-if controls"],
        ["Tests", "Schema, reproducibility, leakage, model, explanation and API tests"],
    ]
    demo_table = Table(demo_rows, colWidths=[1.45 * inch, 5.05 * inch])
    demo_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), BLUE),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.3),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CAD5E0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, GRAY]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(demo_table)

    story.append(Paragraph("8. Limitations", styles["Section"]))
    story.append(
        Paragraph(
            "The primary dataset is synthetic and does not establish production accuracy. The sample size is intentionally limited to the assignment scope. Source reliability, congestion and document status are simulated proxies. Local feature perturbations are explanatory diagnostics, not causal claims. Before deployment, the schema and models should be validated in shadow mode on real SAFiRi event streams, with route-specific calibration and monitoring.",
            styles["BodyCompact"],
        )
    )
    story.append(Paragraph("9. Conclusion", styles["Section"]))
    story.append(
        Paragraph(
            "The prototype demonstrates that ETA prediction can be implemented as an auditable event system rather than an isolated regression. It captures delay propagation through current-stage delay and future stage durations, handles incomplete reporting explicitly, compares against clear baselines and returns an analyst-ready package containing ETA, uncertainty, risk, stage breakdown, data quality, exceptions and actions.",
            styles["BodyCompact"],
        )
    )
    story.append(Spacer(1, 0.08 * inch))
    story.append(
        Paragraph(
            "References: SAFiRi AI Internship Take-Home Assignment; DCSA Track & Trace event concepts; U.S. BTS On-Time Performance as an optional real-data extension.",
            styles["Small"],
        )
    )

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return output
