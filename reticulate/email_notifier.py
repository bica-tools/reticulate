"""Email notifier: sends research updates to the Research Director.

Sends emails to the Research Director (and only to them) when:
- A new step is completed
- A paper draft is produced or updated
- A publication plan entry changes status
- A daily/sprint report is generated

Uses Gmail SMTP with app password. Never sends to anyone else.

Setup:
    export RESEARCH_EMAIL_TO="zua@bica-tools.org"
    export RESEARCH_EMAIL_FROM="your-gmail@gmail.com"
    export RESEARCH_EMAIL_PASSWORD="your-app-password"

Usage:
    from reticulate.email_notifier import notify_step, notify_paper, notify_report
    notify_step("23b", "S/T-invariants", "A+")
    notify_paper("F0", "Session Type State Spaces Form Lattices", "Strong Accept")
    notify_report("Sprint 7 complete. 78/78 at A+.")
"""

from __future__ import annotations

import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dataclasses import dataclass


@dataclass(frozen=True)
class EmailConfig:
    to: str
    from_addr: str
    password: str
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587


def _get_config() -> EmailConfig | None:
    to = os.environ.get("RESEARCH_EMAIL_TO")
    from_addr = os.environ.get("RESEARCH_EMAIL_FROM")
    password = os.environ.get("RESEARCH_EMAIL_PASSWORD")
    if not all([to, from_addr, password]):
        return None
    return EmailConfig(to=to, from_addr=from_addr, password=password)


def _send(subject: str, body_html: str, body_text: str) -> bool:
    """Send an email. Returns True on success."""
    config = _get_config()
    if config is None:
        print("[email_notifier] Not configured. Set RESEARCH_EMAIL_* env vars.")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"[Research] {subject}"
    msg["From"] = config.from_addr
    msg["To"] = config.to

    msg.attach(MIMEText(body_text, "plain"))
    msg.attach(MIMEText(body_html, "html"))

    try:
        with smtplib.SMTP(config.smtp_host, config.smtp_port) as server:
            server.starttls()
            server.login(config.from_addr, config.password)
            server.sendmail(config.from_addr, config.to, msg.as_string())
        return True
    except Exception as e:
        print(f"[email_notifier] Failed: {e}")
        return False


def _research_summary() -> str:
    """Generate a brief research status summary."""
    try:
        from reticulate.evaluator import evaluate_all
        results = evaluate_all(run_tests=False)
        accepted = sum(1 for r in results if r.accepted)
        total = len(results)
        return f"{accepted}/{total} steps at A+"
    except Exception:
        return "(status unavailable)"


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def notify_step(step_number: str, title: str, grade: str) -> bool:
    """Notify that a step was completed or updated."""
    status = _research_summary()
    ts = _timestamp()

    subject = f"Step {step_number} — {grade}"

    body_text = f"""Step {step_number}: {title}
Grade: {grade}
Time: {ts}

Programme: {status}
"""

    body_html = f"""
    <div style="font-family: Arial, sans-serif; max-width: 600px;">
        <h2 style="color: #1565c0;">Step {step_number}: {title}</h2>
        <table style="border-collapse: collapse; width: 100%;">
            <tr><td style="padding: 8px; font-weight: bold;">Grade</td>
                <td style="padding: 8px; color: {'#2e7d32' if grade == 'A+' else '#c62828'};">{grade}</td></tr>
            <tr><td style="padding: 8px; font-weight: bold;">Time</td>
                <td style="padding: 8px;">{ts}</td></tr>
            <tr><td style="padding: 8px; font-weight: bold;">Programme</td>
                <td style="padding: 8px;">{status}</td></tr>
        </table>
    </div>
    """

    return _send(subject, body_html, body_text)


def notify_paper(paper_id: str, title: str, verdict: str, venue: str = "") -> bool:
    """Notify that a paper draft was produced or reviewed."""
    status = _research_summary()
    ts = _timestamp()

    subject = f"Paper {paper_id} — {verdict}"

    body_text = f"""Paper {paper_id}: {title}
Verdict: {verdict}
Venue: {venue or '(not assigned)'}
Time: {ts}

Programme: {status}
"""

    body_html = f"""
    <div style="font-family: Arial, sans-serif; max-width: 600px;">
        <h2 style="color: #1565c0;">Paper {paper_id}: {title}</h2>
        <table style="border-collapse: collapse; width: 100%;">
            <tr><td style="padding: 8px; font-weight: bold;">Verdict</td>
                <td style="padding: 8px; color: {'#2e7d32' if 'Accept' in verdict else '#f57f17'};">{verdict}</td></tr>
            <tr><td style="padding: 8px; font-weight: bold;">Venue</td>
                <td style="padding: 8px;">{venue or '(not assigned)'}</td></tr>
            <tr><td style="padding: 8px; font-weight: bold;">Time</td>
                <td style="padding: 8px;">{ts}</td></tr>
            <tr><td style="padding: 8px; font-weight: bold;">Programme</td>
                <td style="padding: 8px;">{status}</td></tr>
        </table>
    </div>
    """

    return _send(subject, body_html, body_text)


def notify_report(report_text: str, sprint: int = 0) -> bool:
    """Notify with a sprint/daily report."""
    status = _research_summary()
    ts = _timestamp()

    subject = f"Sprint {sprint} Report" if sprint else "Daily Report"

    body_text = f"""{subject}
Time: {ts}
Programme: {status}

{report_text}
"""

    body_html = f"""
    <div style="font-family: Arial, sans-serif; max-width: 600px;">
        <h2 style="color: #1565c0;">{subject}</h2>
        <p style="color: #666;">{ts} — {status}</p>
        <pre style="background: #f5f5f5; padding: 16px; border-radius: 8px; font-size: 13px; overflow-x: auto;">
{report_text}
        </pre>
    </div>
    """

    return _send(subject, body_html, body_text)


def notify_deadline(paper_id: str, title: str, venue: str, days_left: int) -> bool:
    """Notify about an approaching deadline."""
    status = _research_summary()
    ts = _timestamp()
    urgency = "URGENT" if days_left <= 7 else "reminder"

    subject = f"Deadline {urgency}: {venue} in {days_left} days"

    body_text = f"""Deadline {urgency}!
Paper {paper_id}: {title}
Venue: {venue}
Days left: {days_left}
Time: {ts}

Programme: {status}
"""

    body_html = f"""
    <div style="font-family: Arial, sans-serif; max-width: 600px;">
        <h2 style="color: {'#c62828' if days_left <= 7 else '#f57f17'};">
            {'🚨' if days_left <= 7 else '⏰'} {venue} — {days_left} days left
        </h2>
        <p><strong>Paper {paper_id}:</strong> {title}</p>
        <p style="color: #666;">{ts} — {status}</p>
    </div>
    """

    return _send(subject, body_html, body_text)
