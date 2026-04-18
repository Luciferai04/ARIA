# aria/export.py
# Session export — generates a professional PDF from the ARIA research session.

import io
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Table, TableStyle
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY


def generate_session_pdf(history: list, thread_id: str) -> bytes:
    """
    Generate a downloadable PDF summarising the full ARIA research session.

    Args:
        history: list of message dicts from st.session_state.history
        thread_id: the session's unique thread ID

    Returns:
        PDF file content as bytes.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            topMargin=25*mm, bottomMargin=20*mm,
                            leftMargin=20*mm, rightMargin=20*mm)

    styles = getSampleStyleSheet()

    # ── Custom styles ────────────────────────────────────────
    styles.add(ParagraphStyle(
        "CoverTitle", parent=styles["Title"],
        fontSize=28, textColor=HexColor("#6366F1"),
        spaceAfter=8*mm, alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        "CoverSub", parent=styles["Normal"],
        fontSize=12, textColor=HexColor("#94a3b8"),
        alignment=TA_CENTER, spaceAfter=4*mm,
    ))
    styles.add(ParagraphStyle(
        "QuestionHeader", parent=styles["Heading2"],
        fontSize=14, textColor=HexColor("#6366F1"),
        spaceBefore=6*mm, spaceAfter=3*mm,
    ))
    styles.add(ParagraphStyle(
        "ARIABody", parent=styles["Normal"],
        fontSize=10, leading=14, alignment=TA_JUSTIFY,
        spaceAfter=2*mm,
    ))
    styles.add(ParagraphStyle(
        "ARIABullet", parent=styles["Normal"],
        fontSize=10, leading=13, leftIndent=10*mm,
        spaceAfter=1*mm,
    ))
    styles.add(ParagraphStyle(
        "ARIAMeta", parent=styles["Normal"],
        fontSize=9, textColor=HexColor("#94a3b8"),
        spaceAfter=2*mm,
    ))

    story = []

    # ── Cover page ───────────────────────────────────────────
    story.append(Spacer(1, 40*mm))
    story.append(Paragraph("ARIA Research Session", styles["CoverTitle"]))
    story.append(Paragraph("Agentic Research Intelligence Assistant", styles["CoverSub"]))
    story.append(Spacer(1, 10*mm))

    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    assistant_msgs = [m for m in history if m.get("role") == "assistant"]
    total_qs = len(assistant_msgs)

    meta_lines = [
        f"Date: {date_str}",
        f"Session ID: {thread_id}",
        f"Total Questions Answered: {total_qs}",
    ]
    for line in meta_lines:
        story.append(Paragraph(line, styles["CoverSub"]))

    story.append(Spacer(1, 20*mm))
    story.append(HRFlowable(width="80%", thickness=1, color=HexColor("#4a5568")))
    story.append(Spacer(1, 10*mm))

    # ── Q/A pairs ────────────────────────────────────────────
    q_num = 0
    for i, msg in enumerate(history):
        if msg.get("role") == "user":
            q_num += 1
            story.append(Paragraph(f"Q{q_num}: {msg['content']}", styles["QuestionHeader"]))

        elif msg.get("role") == "assistant":
            # Route badge
            route = msg.get("route", "both")
            route_labels = {
                "retrieve": "Knowledge Base Only",
                "tool": "Live Search",
                "both": "KB & Live Search",
                "cache": "Cache Hit",
            }
            story.append(Paragraph(
                f"Route: {route_labels.get(route, route)} | "
                f"Faithfulness: {msg.get('faithfulness', 0.0):.2f}",
                styles["ARIAMeta"]
            ))

            # Summary
            report = msg.get("report", {})
            if isinstance(report, dict):
                summary = report.get("summary", msg.get("content", ""))
            else:
                summary = str(report)
            story.append(Paragraph(summary, styles["ARIABody"]))

            # Key Findings
            findings = report.get("key_findings", []) if isinstance(report, dict) else []
            if findings:
                story.append(Paragraph("<b>Key Findings:</b>", styles["ARIABody"]))
                for j, f in enumerate(findings, 1):
                    story.append(Paragraph(f"{j}. {f}", styles["ARIABullet"]))

            # Sources
            sources = msg.get("sources", [])
            if sources:
                story.append(Paragraph("<b>Sources:</b>", styles["ARIABody"]))
                for j, s in enumerate(sources, 1):
                    story.append(Paragraph(f"[{j}] {s}", styles["ARIABullet"]))

            # Follow-ups
            follow_ups = report.get("follow_ups", []) if isinstance(report, dict) else []
            if follow_ups:
                story.append(Paragraph("<b>Follow-up Questions:</b>", styles["ARIABody"]))
                for q in follow_ups:
                    story.append(Paragraph(f"- {q}", styles["ARIABullet"]))

            story.append(Spacer(1, 4*mm))
            story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor("#374151")))

    doc.build(story)
    return buf.getvalue()
