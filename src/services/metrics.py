"""Metrics tracking service for monitoring system performance."""
import time
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Dict
import structlog

logger = structlog.get_logger(__name__)

@dataclass
class LatencyMetric:
    """Tracks average and max latency."""
    count: int = 0
    total_ms: float = 0.0
    max_ms: float = 0.0

    def record(self, ms: float):
        self.count += 1
        self.total_ms += ms
        self.max_ms = max(self.max_ms, ms)

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.count if self.count > 0 else 0.0

class MetricsService:
    """Service for tracking system performance."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.start_time = datetime.utcnow()
        self.retrieval_latency = LatencyMetric()
        self.generation_latency = LatencyMetric()
        self.total_queries = 0
        self.total_documents = 0  # Added missing counter
        self._data_lock = threading.RLock()

    def record_retrieval(self, ms: float):
        with self._data_lock:
            self.retrieval_latency.record(ms)

    def record_generation(self, ms: float):
        with self._data_lock:
            self.generation_latency.record(ms)

    def record_query(self):
        with self._data_lock:
            self.total_queries += 1

    # --- THIS WAS MISSING ---
    def record_document_processed(self, count: int):
        with self._data_lock:
            self.total_documents += 1
    # ------------------------

    def get_summary(self) -> Dict:
        """Returns a snapshot of system health."""
        with self._data_lock:
            return {
                "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds(),
                "total_queries": self.total_queries,
                "total_documents": self.total_documents,
                "avg_retrieval_ms": round(self.retrieval_latency.avg_ms, 2),
                "avg_generation_ms": round(self.generation_latency.avg_ms, 2),
            }

class Timer:
    """Context manager for easy timing of code blocks."""
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self.start) * 1000

# Singleton
metrics = MetricsService()