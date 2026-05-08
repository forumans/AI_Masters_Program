"""
Monitoring System - The Health Tracker for Multi-Agent System

PURPOSE:
--------
This file provides all the tools to monitor the health and performance
of the multi-agent system. It's like having a dashboard that shows
how well everything is working.

WHAT IT PROVIDES:
-----------------
- Structured logging (organized, searchable logs)
- Performance metrics (how fast things are running)
- Health checks (is everything working?)
- Error tracking (what's going wrong?)
- Resource monitoring (CPU, memory usage)

WHY MONITORING IS IMPORTANT:
----------------------------
- Know when agents are failing
- Track performance over time
- Debug problems with detailed logs
- Ensure system reliability
- Make data-driven improvements

HOW IT WORKS:
------------
1. Logs are written in structured JSON format
2. Metrics track counts, timing, and system state
3. Health checks verify system components
4. Prometheus-compatible metrics for dashboards
5. Real-time monitoring of agent performance
"""

import time
import json
import logging
import structlog
from typing import Any, Dict, Optional, List
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
import asyncio
from contextlib import asynccontextmanager, contextmanager

from prometheus_client import Counter, Histogram, Gauge, start_http_server


def configure_logging(log_level: str = "INFO"):
    """
    Set up structured logging for the entire system.
    
    WHAT IT DOES:
    - Configures logging to output structured JSON
    - Adds timestamps, log levels, and context to all logs
    - Makes logs searchable and easy to parse
    
    WHY STRUCTURED LOGS:
    - Easy to search and filter
    - Can be parsed by monitoring tools
    - Consistent format across all components
    - Better than plain text logs for debugging
    
    Args:
        log_level: How much detail to log (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        format="%(message)s",
        stream=None,  # We'll handle output through structlog
        level=getattr(logging, log_level.upper())
    )
    
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a logger for a specific component.
    
    WHAT IT DOES:
    - Returns a structured logger for the given component name
    - The logger will include context like timestamps and component name
    
    EXAMPLE:
    logger = get_logger("equity_agent")
    logger.info("Analyzing stock", symbol="AAPL")
    # Output: {"timestamp": "...", "level": "info", "component": "equity_agent", "symbol": "AAPL", "message": "Analyzing stock"}
    
    Args:
        name: Name of the component (e.g., "equity_agent", "orchestrator")
        
    Returns:
        Structured logger ready for use
    """
    return structlog.get_logger(name)


# Initialize logging
configure_logging()
logger = get_logger(__name__)


@dataclass
class MetricValue:
    """Single metric value with labels."""
    value: float
    labels: Dict[str, str]
    timestamp: datetime = field(default_factory=datetime.now)


class MetricsCollector:
    """Custom metrics collector with in-memory storage."""
    
    def __init__(self):
        self.counters: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.gauges: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.histograms: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._lock = threading.Lock()
        
        # Prometheus metrics
        self._prometheus_counters = {}
        self._prometheus_gauges = {}
        self._prometheus_histograms = {}
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        labels = labels or {}
        label_key = json.dumps(labels, sort_keys=True)
        
        with self._lock:
            self.counters[name][label_key] += value
            
            # Update Prometheus counter
            if name not in self._prometheus_counters:
                self._prometheus_counters[name] = Counter(
                    name, name, list(labels.keys()) if labels else None
                )
            
            if labels:
                self._prometheus_counters[name].labels(**labels).inc(value)
            else:
                self._prometheus_counters[name].inc(value)
        
        logger.debug("Counter incremented", metric=name, value=value, labels=labels)
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge metric."""
        labels = labels or {}
        label_key = json.dumps(labels, sort_keys=True)
        
        with self._lock:
            self.gauges[name][label_key] = value
            
            # Update Prometheus gauge
            if name not in self._prometheus_gauges:
                self._prometheus_gauges[name] = Gauge(
                    name, name, list(labels.keys()) if labels else None
                )
            
            if labels:
                self._prometheus_gauges[name].labels(**labels).set(value)
            else:
                self._prometheus_gauges[name].set(value)
        
        logger.debug("Gauge set", metric=name, value=value, labels=labels)
    
    def histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a histogram value."""
        labels = labels or {}
        label_key = json.dumps(labels, sort_keys=True)
        
        with self._lock:
            self.histograms[name].append(MetricValue(value, labels))
            
            # Update Prometheus histogram
            if name not in self._prometheus_histograms:
                self._prometheus_histograms[name] = Histogram(
                    name, name, list(labels.keys()) if labels else None
                )
            
            if labels:
                self._prometheus_histograms[name].labels(**labels).observe(value)
            else:
                self._prometheus_histograms[name].observe(value)
        
        logger.debug("Histogram recorded", metric=name, value=value, labels=labels)
    
    def get_counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """Get counter value."""
        labels = labels or {}
        label_key = json.dumps(labels, sort_keys=True)
        return self.counters[name].get(label_key, 0.0)
    
    def get_gauge(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """Get gauge value."""
        labels = labels or {}
        label_key = json.dumps(labels, sort_keys=True)
        return self.gauges[name].get(label_key, 0.0)
    
    def get_histogram_stats(self, name: str, labels: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """Get histogram statistics."""
        labels = labels or {}
        values = [
            mv.value for mv in self.histograms[name]
            if mv.labels == labels
        ]
        
        if not values:
            return {}
        
        values.sort()
        count = len(values)
        
        return {
            'count': count,
            'sum': sum(values),
            'avg': sum(values) / count,
            'min': values[0],
            'max': values[-1],
            'p50': values[int(count * 0.5)],
            'p95': values[int(count * 0.95)],
            'p99': values[int(count * 0.99)],
        }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics as a dictionary."""
        with self._lock:
            return {
                'counters': {
                    name: dict(values) for name, values in self.counters.items()
                },
                'gauges': {
                    name: dict(values) for name, values in self.gauges.items()
                },
                'histograms': {
                    name: {
                        json.dumps(labels, sort_keys=True): self.get_histogram_stats(name, labels)
                        for labels in set(mv.labels for mv in values)
                    }
                    for name, values in self.histograms.items()
                }
            }
    
    def reset_metrics(self):
        """Reset all metrics."""
        with self._lock:
            self.counters.clear()
            self.gauges.clear()
            self.histograms.clear()
        
        logger.info("All metrics reset")


# Global metrics collector
metrics = MetricsCollector()


class HealthChecker:
    """Health check manager for system components."""
    
    def __init__(self):
        self.checks: Dict[str, callable] = {}
        self.last_results: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
    
    def register_check(self, name: str, check_func: callable):
        """Register a health check function."""
        self.checks[name] = check_func
        logger.info("Health check registered", check=name)
    
    async def run_check(self, name: str) -> Dict[str, Any]:
        """Run a specific health check."""
        if name not in self.checks:
            return {
                'status': 'unhealthy',
                'message': f'Health check {name} not found',
                'timestamp': datetime.now().isoformat()
            }
        
        start_time = time.time()
        try:
            if asyncio.iscoroutinefunction(self.checks[name]):
                result = await self.checks[name]()
            else:
                result = self.checks[name]()
            
            duration = time.time() - start_time
            
            health_result = {
                'status': 'healthy' if result else 'unhealthy',
                'message': str(result) if result else 'OK',
                'duration': duration,
                'timestamp': datetime.now().isoformat()
            }
            
            metrics.histogram('health_check_duration', duration, {'check': name})
            metrics.increment_counter('health_check_total', 1, {'check': name, 'status': health_result['status']})
            
        except Exception as e:
            duration = time.time() - start_time
            health_result = {
                'status': 'unhealthy',
                'message': str(e),
                'duration': duration,
                'timestamp': datetime.now().isoformat()
            }
            
            metrics.increment_counter('health_check_total', 1, {'check': name, 'status': 'unhealthy'})
            logger.error("Health check failed", check=name, error=str(e))
        
        with self._lock:
            self.last_results[name] = health_result
        
        return health_result
    
    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks."""
        results = {}
        overall_status = 'healthy'
        
        for name in self.checks:
            results[name] = await self.run_check(name)
            if results[name]['status'] != 'healthy':
                overall_status = 'unhealthy'
        
        return {
            'status': overall_status,
            'checks': results,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_last_results(self) -> Dict[str, Any]:
        """Get last health check results."""
        with self._lock:
            return self.last_results.copy()


# Global health checker
health_checker = HealthChecker()


@asynccontextmanager
async def monitor_execution(operation: str, labels: Optional[Dict[str, str]] = None):
    """Context manager to monitor operation execution."""
    labels = labels or {}
    start_time = time.time()
    
    metrics.increment_counter('operations_started', 1, {'operation': operation, **labels})
    
    try:
        logger.info("Operation started", operation=operation, labels=labels)
        yield
        duration = time.time() - start_time
        
        metrics.histogram('operation_duration', duration, {'operation': operation, **labels})
        metrics.increment_counter('operations_completed', 1, {'operation': operation, **labels})
        
        logger.info("Operation completed", operation=operation, duration=duration, labels=labels)
    
    except Exception as e:
        duration = time.time() - start_time
        
        metrics.histogram('operation_duration', duration, {'operation': operation, **labels, 'status': 'error'})
        metrics.increment_counter('operations_failed', 1, {'operation': operation, **labels, 'error': type(e).__name__})
        
        logger.error("Operation failed", operation=operation, duration=duration, error=str(e), labels=labels)
        raise


@contextmanager
def monitor_execution_sync(operation: str, labels: Optional[Dict[str, str]] = None):
    """Synchronous context manager to monitor operation execution."""
    labels = labels or {}
    start_time = time.time()
    
    metrics.increment_counter('operations_started', 1, {'operation': operation, **labels})
    
    try:
        logger.info("Operation started", operation=operation, labels=labels)
        yield
        duration = time.time() - start_time
        
        metrics.histogram('operation_duration', duration, {'operation': operation, **labels})
        metrics.increment_counter('operations_completed', 1, {'operation': operation, **labels})
        
        logger.info("Operation completed", operation=operation, duration=duration, labels=labels)
    
    except Exception as e:
        duration = time.time() - start_time
        
        metrics.histogram('operation_duration', duration, {'operation': operation, **labels, 'status': 'error'})
        metrics.increment_counter('operations_failed', 1, {'operation': operation, **labels, 'error': type(e).__name__})
        
        logger.error("Operation failed", operation=operation, duration=duration, error=str(e), labels=labels)
        raise


class PerformanceTracker:
    """Track performance metrics over time."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.response_times: deque = deque(maxlen=window_size)
        self.error_rates: deque = deque(maxlen=window_size)
        self.throughput: deque = deque(maxlen=window_size)
        self._lock = threading.Lock()
    
    def record_request(self, response_time: float, success: bool):
        """Record a request."""
        with self._lock:
            self.response_times.append(response_time)
            self.error_rates.append(0 if success else 1)
            self.throughput.append(1)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        with self._lock:
            if not self.response_times:
                return {}
            
            response_times = list(self.response_times)
            error_rate = sum(self.error_rates) / len(self.error_rates) if self.error_rates else 0
            throughput = sum(self.throughput) / 60 if self.throughput else 0  # requests per second
            
            response_times.sort()
            count = len(response_times)
            
            return {
                'requests_per_minute': throughput * 60,
                'error_rate': error_rate,
                'response_time': {
                    'avg': sum(response_times) / count,
                    'p50': response_times[int(count * 0.5)],
                    'p95': response_times[int(count * 0.95)],
                    'p99': response_times[int(count * 0.99)],
                    'max': max(response_times),
                    'min': min(response_times)
                }
            }


# Global performance tracker
performance_tracker = PerformanceTracker()


def start_metrics_server(port: int = 9090):
    """Start Prometheus metrics server."""
    try:
        start_http_server(port)
        logger.info("Metrics server started", port=port)
    except Exception as e:
        logger.error("Failed to start metrics server", error=str(e))


# Default health checks
async def check_openai_api():
    """Check OpenAI API connectivity."""
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-3.5-turbo", max_tokens=1)
        await llm.ainvoke("test")
        return True
    except Exception as e:
        return f"OpenAI API check failed: {str(e)}"


def check_memory_usage():
    """Check memory usage."""
    import psutil
    memory = psutil.virtual_memory()
    if memory.percent > 90:
        return f"High memory usage: {memory.percent}%"
    return True


# Register default health checks
health_checker.register_check("openai_api", check_openai_api)
health_checker.register_check("memory", check_memory_usage)
