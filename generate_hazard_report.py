"""
Builds hazard-report.pdf from the events this run just produced (all hazard
classes — Pothole, water_logging, debris, garbage, Accident, etc.), not just
accidents. Only called when there IS at least one event — see
detect_hazard.py, which skips this entirely on a clean video.

Each record embeds the SAME frame image that extract_frames.py already
saved into FRAMES_DIR with time/lat/lon/bus/zone burned into the pixels —
no separate annotated copy, no bounding box drawing.
"""

import os
import json
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
)

from config import REPORTS_DIR, HAZARD_REPORT_PDF, PLATES_OUTPUT_JSON, FRAMES_DIR


def load_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def _plates_for(event_id, plates):
    return [p for p in plates if p.get("accident_event_id") == event_id]


def build_rows(events, plates):
    rows = []
    for event in events:
        location = event.get("location") or {}
        matched_plates = _plates_for(event["event_id"], plates) if event["class"] == "Accident" else []

        base = {
            "event_id": event["event_id"],
            "class": event["class"],
            "confidence": f"{event.get('confidence', 0) * 100:.1f}%",
            "timestamp": event.get("timestamp", "-"),
            "latitude": location.get("lat", "-"),
            "longitude": location.get("lon", "-"),
            "bus_id": location.get("bus_id", "-"),
            "place": location.get("zone", "-"),
            "frame_filename": event.get("frame_filename"),
        }

        if not matched_plates:
            rows.append({**base, "plate_number": "-", "ocr_confidence": "-"})
        else:
            for plate in matched_plates:
                rows.append({
                    **base,
                    "plate_number": plate.get("plate_number", "-"),
                    "ocr_confidence": f"{plate.get('ocr_confidence', 0):.1f}%",
                })

    rows.sort(key=lambda r: r["timestamp"] or "")
    return rows


def generate_hazard_report(events, frames_dir=FRAMES_DIR, plates_path=PLATES_OUTPUT_JSON):
    """Returns the path to the generated PDF, or None if there was nothing
    to report (caller already checks `events` is non-empty, but this stays
    defensive in case it's called directly)."""
    if not events:
        print("No hazard events — skipping PDF report.")
        return None

    plates = load_json(plates_path, [])
    rows = build_rows(events, plates)
    if not rows:
        print("No hazard rows to report — skipping PDF report.")
        return None

    os.makedirs(REPORTS_DIR, exist_ok=True)
    output_pdf = os.path.join(REPORTS_DIR, HAZARD_REPORT_PDF)

    doc = SimpleDocTemplate(
        output_pdf, pagesize=letter,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
    )
    styles = getSampleStyleSheet()
    label_style = ParagraphStyle("label", parent=styles["Normal"], fontSize=9,
                                  fontName="Helvetica-Bold", textColor=colors.black)
    value_style = ParagraphStyle("value", parent=styles["Normal"], fontSize=9, textColor=colors.black)
    heading_style = ParagraphStyle("heading", parent=styles["Heading2"], fontSize=11,
                                    textColor=colors.black, spaceBefore=4, spaceAfter=6)
    caption_style = ParagraphStyle("caption", parent=styles["Normal"], fontSize=8,
                                    textColor=colors.grey)

    story = [
        Paragraph("Road Hazard Detection Report", styles["Title"]),
        Paragraph(
            f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &nbsp;|&nbsp; "
            f"{len(rows)} record(s)", styles["Normal"],
        ),
        Spacer(1, 16),
    ]

    field_labels = [
        ("Event ID", "event_id"),
        ("Class Detected", "class"),
        ("Confidence", "confidence"),
        ("Timestamp", "timestamp"),
        ("Latitude", "latitude"),
        ("Longitude", "longitude"),
        ("Bus ID", "bus_id"),
        ("Place / Zone", "place"),
        ("Plate Number", "plate_number"),
        ("Plate OCR Confidence", "ocr_confidence"),
    ]

    for i, r in enumerate(rows, start=1):
        story.append(Paragraph(f"Record {i} — {r['class']}", heading_style))

        table_data = [
            [Paragraph(label, label_style), Paragraph(str(r[key]), value_style)]
            for label, key in field_labels
            if not (key in ("plate_number", "ocr_confidence") and r["plate_number"] == "-" and r["class"] != "Accident")
        ]
        table = Table(table_data, colWidths=[2.2 * inch, 3.8 * inch])
        table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.75, colors.black),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(table)
        story.append(Spacer(1, 8))

        frame_path = os.path.join(frames_dir, r["frame_filename"]) if r["frame_filename"] else None
        if frame_path and os.path.exists(frame_path):
            story.append(Image(frame_path, width=4.2 * inch, height=2.6 * inch))
            story.append(Paragraph("Detected frame (time/location already overlaid)", caption_style))
        story.append(Spacer(1, 20))

    doc.build(story)
    print(f"Saved hazard report -> {output_pdf} ({len(rows)} row(s))")
    return output_pdf


if __name__ == "__main__":
    events = load_json(os.path.join("events", "events.json"), [])
    generate_hazard_report(events)