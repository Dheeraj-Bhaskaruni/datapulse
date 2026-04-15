"""Simple task scheduler for periodic operations."""

import logging
import time
import threading
from typing import Callable, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ScheduledTask:
    name: str
    func: Callable
    interval_seconds: int
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    enabled: bool = True
    kwargs: Dict[str, Any] = field(default_factory=dict)


class TaskScheduler:
    """Simple task scheduler for periodic retraining and monitoring."""

    def __init__(self):
        self.tasks: Dict[str, ScheduledTask] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def add_task(self, name: str, func: Callable, interval_seconds: int, **kwargs) -> None:
        """Add a scheduled task."""
        task = ScheduledTask(
            name=name,
            func=func,
            interval_seconds=interval_seconds,
            next_run=datetime.now(),
            kwargs=kwargs,
        )
        self.tasks[name] = task
        logger.info(f"Scheduled task '{name}' every {interval_seconds}s")

    def remove_task(self, name: str) -> None:
        """Remove a scheduled task."""
        if name in self.tasks:
            del self.tasks[name]

    def _run_loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            now = datetime.now()
            for task in self.tasks.values():
                if task.enabled and task.next_run and now >= task.next_run:
                    try:
                        logger.info(f"Running task: {task.name}")
                        task.func(**task.kwargs)
                        task.last_run = now
                        task.run_count += 1
                    except Exception as e:
                        logger.error(f"Task '{task.name}' failed: {e}")
                    task.next_run = now + timedelta(seconds=task.interval_seconds)
            time.sleep(1)

    def start(self) -> None:
        """Start the scheduler in a background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("Scheduler started")

    def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Scheduler stopped")

    def status(self) -> Dict[str, Any]:
        """Get scheduler status."""
        return {
            'running': self._running,
            'tasks': {
                name: {
                    'enabled': t.enabled,
                    'run_count': t.run_count,
                    'last_run': str(t.last_run),
                    'interval': t.interval_seconds,
                }
                for name, t in self.tasks.items()
            },
        }
