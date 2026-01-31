"""
Notification module for INSIGHT Web Application.

Provides notification functionality via:
- Email
- Webhooks
- In-app notifications
"""

import asyncio
import hashlib
import hmac
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp

from web.database import ResearchSession


@dataclass
class Notification:
    """Notification model."""
    id: str
    user_id: int
    type: str
    title: str
    message: str
    data: Dict[str, Any]
    read: bool = False
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


class NotificationManager:
    """
    Manages notifications across different channels.
    """

    def __init__(self):
        """Initialize notification manager."""
        self._email_config = {
            "smtp_host": os.environ.get("SMTP_HOST", ""),
            "smtp_port": int(os.environ.get("SMTP_PORT", "587")),
            "smtp_user": os.environ.get("SMTP_USER", ""),
            "smtp_password": os.environ.get("SMTP_PASSWORD", ""),
            "from_email": os.environ.get("FROM_EMAIL", "noreply@insight.ai")
        }

    async def notify_completion(
        self,
        session: ResearchSession,
        webhooks: List[Dict] = None,
        email: str = None
    ) -> None:
        """
        Send completion notification.

        Args:
            session: Completed session
            webhooks: Webhook configurations to trigger
            email: Email address for notification
        """
        # Prepare notification data
        data = {
            "event": "research_completed",
            "session_id": session.id,
            "objective": session.objective,
            "status": session.status,
            "results_count": session.results_count,
            "completed_at": datetime.utcnow().isoformat()
        }

        tasks = []

        # Trigger webhooks
        if webhooks:
            for webhook in webhooks:
                if "completed" in webhook.get("events", []):
                    tasks.append(
                        self._trigger_webhook(webhook, data)
                    )

        # Send email
        if email:
            tasks.append(
                self._send_email(
                    to=email,
                    subject=f"INSIGHT Research Completed: {session.objective[:50]}",
                    body=self._format_completion_email(session)
                )
            )

        # Execute all notifications concurrently
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def notify_failure(
        self,
        session: ResearchSession,
        error: str,
        webhooks: List[Dict] = None,
        email: str = None
    ) -> None:
        """
        Send failure notification.

        Args:
            session: Failed session
            error: Error message
            webhooks: Webhook configurations
            email: Email address
        """
        data = {
            "event": "research_failed",
            "session_id": session.id,
            "objective": session.objective,
            "error": error,
            "failed_at": datetime.utcnow().isoformat()
        }

        tasks = []

        if webhooks:
            for webhook in webhooks:
                if "failed" in webhook.get("events", []):
                    tasks.append(
                        self._trigger_webhook(webhook, data)
                    )

        if email:
            tasks.append(
                self._send_email(
                    to=email,
                    subject=f"INSIGHT Research Failed: {session.objective[:50]}",
                    body=f"Research session failed.\n\nError: {error}"
                )
            )

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _trigger_webhook(
        self,
        webhook: Dict,
        data: Dict
    ) -> bool:
        """
        Trigger a webhook.

        Args:
            webhook: Webhook configuration
            data: Data to send

        Returns:
            True if successful
        """
        url = webhook.get("url")
        secret = webhook.get("secret")

        if not url:
            return False

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "INSIGHT-Webhook/1.0"
        }

        # Add signature if secret is configured
        if secret:
            payload = json.dumps(data)
            signature = hmac.new(
                secret.encode(),
                payload.encode(),
                hashlib.sha256
            ).hexdigest()
            headers["X-Webhook-Signature"] = f"sha256={signature}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=data,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    return response.status < 400

        except Exception as e:
            print(f"Webhook failed: {e}")
            return False

    async def _send_email(
        self,
        to: str,
        subject: str,
        body: str,
        html: str = None
    ) -> bool:
        """
        Send an email notification.

        Args:
            to: Recipient email
            subject: Email subject
            body: Plain text body
            html: Optional HTML body

        Returns:
            True if successful
        """
        if not self._email_config["smtp_host"]:
            print("Email not configured")
            return False

        try:
            import aiosmtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self._email_config["from_email"]
            msg["To"] = to

            msg.attach(MIMEText(body, "plain"))
            if html:
                msg.attach(MIMEText(html, "html"))

            await aiosmtplib.send(
                msg,
                hostname=self._email_config["smtp_host"],
                port=self._email_config["smtp_port"],
                username=self._email_config["smtp_user"],
                password=self._email_config["smtp_password"],
                start_tls=True
            )

            return True

        except ImportError:
            print("aiosmtplib not installed")
            return False
        except Exception as e:
            print(f"Email failed: {e}")
            return False

    def _format_completion_email(self, session: ResearchSession) -> str:
        """Format completion email body."""
        return f"""
Your INSIGHT research has completed!

Objective: {session.objective}

Status: {session.status}
Results: {session.results_count}
Tools Used: {', '.join(session.tools)}

Key Findings:
{session.key_findings[:2000] if session.key_findings else 'No key findings available.'}

View full results at: https://insight.ai/research/{session.id}

---
This is an automated message from INSIGHT AI.
        """.strip()


class InAppNotificationStore:
    """
    In-memory store for in-app notifications.

    In production, this should be backed by a database.
    """

    def __init__(self):
        """Initialize store."""
        self._notifications: Dict[int, List[Notification]] = {}

    def add(self, notification: Notification) -> None:
        """Add a notification."""
        if notification.user_id not in self._notifications:
            self._notifications[notification.user_id] = []

        self._notifications[notification.user_id].insert(0, notification)

        # Keep only last 100 notifications
        self._notifications[notification.user_id] = \
            self._notifications[notification.user_id][:100]

    def get_unread(self, user_id: int) -> List[Notification]:
        """Get unread notifications for user."""
        return [
            n for n in self._notifications.get(user_id, [])
            if not n.read
        ]

    def get_all(self, user_id: int, limit: int = 50) -> List[Notification]:
        """Get all notifications for user."""
        return self._notifications.get(user_id, [])[:limit]

    def mark_read(self, user_id: int, notification_id: str) -> None:
        """Mark notification as read."""
        for notification in self._notifications.get(user_id, []):
            if notification.id == notification_id:
                notification.read = True
                break

    def mark_all_read(self, user_id: int) -> None:
        """Mark all notifications as read."""
        for notification in self._notifications.get(user_id, []):
            notification.read = True

    def get_unread_count(self, user_id: int) -> int:
        """Get count of unread notifications."""
        return len(self.get_unread(user_id))
