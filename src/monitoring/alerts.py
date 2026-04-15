"""Alert system for monitoring anomalies and drift."""

import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    level: AlertLevel
    source: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    acknowledged: bool = False


class AlertManager:
    """Manages alerts for monitoring systems."""

    def __init__(self):
        self.alerts: List[Alert] = []
        self.handlers: Dict[AlertLevel, List[Callable]] = {
            AlertLevel.INFO: [],
            AlertLevel.WARNING: [],
            AlertLevel.CRITICAL: [],
        }

    def add_handler(self, level: AlertLevel, handler: Callable) -> None:
        self.handlers[level].append(handler)

    def trigger(self, level: AlertLevel, source: str, message: str, details: Dict = None) -> Alert:
        """Trigger a new alert."""
        alert = Alert(level=level, source=source, message=message, details=details or {})
        self.alerts.append(alert)

        log_fn = {'info': logger.info, 'warning': logger.warning, 'critical': logger.critical}
        log_fn.get(level.value, logger.info)(f"[{level.value.upper()}] {source}: {message}")

        for handler in self.handlers.get(level, []):
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")

        return alert

    def get_alerts(
        self, level: Optional[AlertLevel] = None, unacknowledged_only: bool = False
    ) -> List[Alert]:
        """Get filtered alerts."""
        alerts = self.alerts
        if level:
            alerts = [a for a in alerts if a.level == level]
        if unacknowledged_only:
            alerts = [a for a in alerts if not a.acknowledged]
        return alerts

    def acknowledge(self, index: int) -> None:
        if 0 <= index < len(self.alerts):
            self.alerts[index].acknowledged = True

    def summary(self) -> Dict[str, int]:
        return {level.value: len([a for a in self.alerts if a.level == level]) for level in AlertLevel}

    def clear(self) -> None:
        self.alerts.clear()
