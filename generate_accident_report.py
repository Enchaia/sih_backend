"""
Builds acc-reports.pdf from the two JSON outputs the pipeline already
produces:

    events/events.json   -> confirmed hazard events (has the ACCIDENT
                             DETECTION confidence, timestamp, location)
    plates.json           -> ANPR plate reads (has the PLATE NUMBER and
                             OCR confidence, linked back via
                             accident_event_id)

These two files don't share every field you asked for on their own —
accident-detection confidence lives on the event, plate number + OCR
confidence live on the plate reading — so this script joins them on
event_id / accident_event_id before laying out the PDF.

Run standalone:
    python generate_accident_report.py

Or import and call generate_report() from detect_hazard.py right after
save_plates(all_plates) to have acc-reports.pdf regenerate on every run.
"""

import os
import json
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
)

from config import EVENTS_DIR, PLATES_OUTPUT_JSON

REPORT_PDF = "acc-reports.pdf"


def load_json(path, default):
    if not os.path.exists(path):
        print(f"Warning: {path} not found, using empty data.")
        return default
    with open(path) as f:
        return json.load(f)


def build_rows(events, plates):
    """One row per (accident event, plate reading) pair. Accidents with
    no matched plate reading still get one row, with plate fields blank,
    so the report doesn't silently drop them."""
    events_by_id = {e["event_id"]: e for e in events if e.get("class") == "Accident"}

    plates_by_event = {}
    for p in plates:
        plates_by_event.setdefault(p.get("accident_event_id"), []).append(p)

    rows = []
    for event_id, event in events_by_id.items():
        location = event.get("location") or {}
        matched_plates = plates_by_event.get(event_id, [])

        if not matched_plates:
            rows.append({
                "accident_event_id": event_id,
                "frame": event.get("representative_frame", "-"),
                "timestamp": event.get("timestamp", "-"),
                "lat": location.get("lat", "-"),
                "lon": location.get("lon", "-"),
                "zone": location.get("zone", "-"),
                "plate_number": "No plate detected",
                "ocr_confidence": "-",
                "accident_confidence": f"{event.get('confidence', 0) * 100:.1f}%",
            })
            continue

        for plate in matched_plates:
            rows.append({
                "accident_event_id": event_id,
                "frame": plate.get("source_frame") or event.get("representative_frame", "-"),
                "timestamp": plate.get("timestamp") or event.get("timestamp", "-"),
                "lat": location.get("lat", "-"),
                "lon": location.get("lon", "-"),
                "zone": location.get("zone", "-"),
                "plate_number": plate.get("plate_number", "-"),
                "ocr_confidence": f"{plate.get('ocr_confidence', 0):.1f}%",
                "accident_confidence": f"{event.get('confidence', 0) * 100:.1f}%",
            })

    rows.sort(key=lambda r: r["timestamp"] or "")
    return rows


def generate_report(events_path=None, plates_path=None, output_pdf=REPORT_PDF):
    """Builds acc-reports.pdf as a stack of vertical (field/value) tables —
    one table per accident+plate record — in plain black-and-white."""
    events_path = events_path or os.path.join(EVENTS_DIR, "events.json")
    plates_path = plates_path or PLATES_OUTPUT_JSON

    events = load_json(events_path, [])
    plates = load_json(plates_path, [])
    rows = build_rows(events, plates)

    doc = SimpleDocTemplate(
        output_pdf,
        pagesize=letter,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
    )
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    meta_style = ParagraphStyle(
        "meta", parent=styles["Normal"], fontSize=9, textColor=colors.black,
    )
    label_style = ParagraphStyle(
        "label", parent=styles["Normal"], fontSize=9, fontName="Helvetica-Bold",
        textColor=colors.black,
    )
    value_style = ParagraphStyle(
        "value", parent=styles["Normal"], fontSize=9, textColor=colors.black,
    )
    record_heading_style = ParagraphStyle(
        "record_heading", parent=styles["Heading2"], fontSize=11,
        textColor=colors.black, spaceBefore=4, spaceAfter=6,
    )

    story = [
        Paragraph("Accident Detection & Plate Recognition Report", title_style),
        Paragraph(
            f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
            f"&nbsp;|&nbsp; {len(rows)} record(s) across "
            f"{len({r['accident_event_id'] for r in rows})} accident event(s)",
            meta_style,
        ),
        Spacer(1, 16),
    ]

    if not rows:
        story.append(Paragraph("No accident events found.", styles["Normal"]))

    field_labels = [
        ("Accident Event ID", "accident_event_id"),
        ("Frame ID", "frame"),
        ("Timestamp", "timestamp"),
        ("Latitude", "lat"),
        ("Longitude", "lon"),
        ("Zone", "zone"),
        ("Plate Number", "plate_number"),
        ("OCR Confidence", "ocr_confidence"),
        ("Accident Detection Confidence", "accident_confidence"),
    ]

    for i, r in enumerate(rows, start=1):
        story.append(Paragraph(f"Record {i}", record_heading_style))

        table_data = [
            [Paragraph(label, label_style), Paragraph(str(r[key]), value_style)]
            for label, key in field_labels
        ]

        table = Table(table_data, colWidths=[2.2 * inch, 3.8 * inch])
        table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.75, colors.black),
            ("BACKGROUND", (0, 0), (-1, -1), colors.white),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(table)
        story.append(Spacer(1, 18))

    doc.build(story)
    print(f"Saved report -> {output_pdf} ({len(rows)} row(s))")


if __name__ == "__main__":
    generate_report()
