"""
Resilience Patterns for Multi-Agent System

PURPOSE:
--------
This file protects our AI agents from failing and keeps the system stable when things go wrong.
It handles common problems like:
- Agents not responding (timeouts)
- Too many requests at once (overload)
- Temporary failures (network issues)
- Agents constantly failing (circuit breakers)

WHAT IT DOES:
-------------
1. Circuit Breakers: Stop calling agents that keep failing
2. Retry Logic: Try again when things fail temporarily  
3. Timeouts: Don't wait forever for slow responses
4. Bulkheads: Limit how many requests run at the same time

HOW TO USE:
-----------
Add @resilient() decorator to any agent function to protect it automatically.
Or use specific decorators like @with_circuit_breaker(), @retry_with_policy(), etc.
"""

import time
import asyncio
import logging
from typing import Any, Callable, Dict, Optional, Union, TypeVar, Callable
from functools import wraps
from enum import Enum
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pybreaker
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    RetryError
)

from .monitoring import metrics, get_logger

logger = get_logger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """
    Circuit Breaker States - Like a traffic light for API calls
    
    CLOSED: Green light - Everything working normally, calls go through
    OPEN: Red light - Agent is failing, block all calls to prevent more damage  
    HALF_OPEN: Yellow light - Testing if agent recovered, allow few calls to check
    """
    CLOSED = "closed"      # Normal operation - calls allowed
    OPEN = "open"          # Agent failing - block all calls
    HALF_OPEN = "half_open"  # Testing recovery - allow limited calls


@dataclass
class CircuitBreakerConfig:
    """
    Settings for Circuit Breaker - When to trip and when to reset
    
    failure_threshold: How many failures before opening circuit (default: 5)
    recovery_timeout: How long to wait before trying again (default: 60 seconds)
    expected_exception: What type of error counts as failure (default: any Exception)
    name: Unique name for this circuit breaker (for logging/metrics)
    """
    failure_threshold: int = 5          # Open circuit after 5 failures
    recovery_timeout: int = 60          # Wait 60 seconds before retry
    expected_exception: type = Exception # Count any exception as failure
    name: str = "default"               # Name for identification


class CircuitBreaker:
    """
    Circuit Breaker - Protects failing agents by stopping calls to them
    
    WHAT IT DOES:
    - Counts failures and opens circuit when too many failures occur
    - Stops all calls when open to prevent cascading failures
    - Tries to recover after waiting period
    - Tracks success/failure metrics for monitoring
    
    HOW IT WORKS:
    1. Start in CLOSED state (allow all calls)
    2. Count failures, open circuit after threshold reached
    3. In OPEN state, reject all calls immediately
    4. After recovery timeout, try HALF_OPEN state (allow few calls)
    5. If calls succeed in HALF_OPEN, go back to CLOSED
    6. If calls fail in HALF_OPEN, go back to OPEN
    """
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.failure_count = 0                    # How many failures so far
        self.last_failure_time: Optional[datetime] = None  # When last failure happened
        self.state = CircuitState.CLOSED          # Start closed (allow calls)
        self._lock = asyncio.Lock()               # Prevent race conditions
        
        # Track performance metrics
        self.metrics = {
            'calls_total': 0,          # Total calls attempted
            'calls_success': 0,        # Successful calls
            'calls_failure': 0,        # Failed calls
            'circuit_open_count': 0    # How many times circuit opened
        }
    
    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute a function with circuit breaker protection
        
        WHAT IT DOES:
        - Checks if circuit is open (reject call if so)
        - Executes the function if circuit allows it
        - Tracks success/failure and updates state
        - Handles both async and sync functions
        
        HOW IT WORKS:
        1. Lock to prevent race conditions
        2. Check circuit state
        3. If OPEN, either reject or try HALF_OPEN
        4. Execute function
        5. Update state based on result
        """
        async with self._lock:
            self.metrics['calls_total'] += 1
            
            # Check if circuit is currently OPEN (blocking calls)
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    # Try recovery - move to HALF_OPEN to test
                    self.state = CircuitState.HALF_OPEN
                    logger.info(f"Circuit breaker {self.config.name} entering HALF_OPEN state")
                else:
                    # Still in recovery period - reject the call
                    self.metrics['calls_failure'] += 1
                    raise CircuitBreakerError(f"Circuit breaker {self.config.name} is OPEN")
        
        try:
            # Execute the function (async or sync)
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            self._on_success()  # Handle successful execution
            return result
        except self.config.expected_exception as e:
            self._on_failure()  # Handle failure
            raise
    
    def _should_attempt_reset(self) -> bool:
        """
        Check if enough time has passed to try recovering
        
        WHAT IT DOES:
        - Checks if we've waited long enough since last failure
        - Returns True if recovery timeout has passed, False otherwise
        
        HOW IT WORKS:
        - If no failures yet, return False (nothing to recover from)
        - Compare current time with last failure time + recovery timeout
        - Return True if enough time has passed for recovery attempt
        """
        if self.last_failure_time is None:
            return False  # No failures yet, nothing to recover
        return datetime.now() - self.last_failure_time > timedelta(seconds=self.config.recovery_timeout)
    
    def _on_success(self):
        """
        Handle successful function call
        
        WHAT IT DOES:
        - Resets failure count (success means we're recovering)
        - If in HALF_OPEN, move back to CLOSED (full recovery)
        - Updates success metrics
        - Logs the recovery for monitoring
        
        HOW IT WORKS:
        1. Clear failure count (start fresh)
        2. If we were testing recovery (HALF_OPEN), now we're fully recovered (CLOSED)
        3. Track success metrics for monitoring
        4. Log the state change for debugging
        """
        self.failure_count = 0  # Success means reset failure count
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED  # Full recovery!
            logger.info(f"Circuit breaker {self.config.name} reset to CLOSED")
        self.metrics['calls_success'] += 1
        metrics.increment_counter('circuit_breaker_success', {'name': self.config.name})
    
    def _on_failure(self):
        """
        Handle failed function call
        
        WHAT IT DOES:
        - Increments failure count
        - Records when failure happened
        - Opens circuit if too many failures
        - Updates failure metrics
        
        HOW IT WORKS:
        1. Count this failure
        2. Remember when it happened (for recovery timing)
        3. If we hit the failure threshold, open the circuit
        4. Track metrics and log for monitoring
        """
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        self.metrics['calls_failure'] += 1
        
        # Check if we've failed too many times and need to open circuit
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN  # Stop all calls!
            self.metrics['circuit_open_count'] += 1
            logger.warning(f"Circuit breaker {self.config.name} opened after {self.failure_count} failures")
            metrics.increment_counter('circuit_breaker_opened', {'name': self.config.name})
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get current circuit breaker status for monitoring
        
        WHAT IT DOES:
        - Returns current state and statistics
        - Used for health checks and debugging
        
        RETURNS:
        - Current state (CLOSED/OPEN/HALF_OPEN)
        - Failure count
        - Last failure time
        - Performance metrics
        """
        return {
            'state': self.state.value,
            'failure_count': self.failure_count,
            'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None,
            'metrics': self.metrics.copy()
        }
    
    def reset(self):
        """
        Manually reset circuit breaker to working state
        
        WHAT IT DOES:
        - Forces circuit breaker back to CLOSED state
        - Clears all failure history
        - Used for manual recovery or testing
        
        HOW IT WORKS:
        1. Reset failure counter to 0
        2. Set state to CLOSED (allow calls)
        3. Clear last failure time
        4. Log the manual reset
        """
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        self.last_failure_time = None
        logger.info(f"Circuit breaker {self.config.name} manually reset")


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    pass


# Pre-configured circuit breakers for different agents
circuit_breakers: Dict[str, CircuitBreaker] = {
    'equity_expert': CircuitBreaker(CircuitBreakerConfig(
        name='equity_expert',
        failure_threshold=3,
        recovery_timeout=30
    )),
    'tax_expert': CircuitBreaker(CircuitBreakerConfig(
        name='tax_expert',
        failure_threshold=3,
        recovery_timeout=30
    )),
    'risk_expert': CircuitBreaker(CircuitBreakerConfig(
        name='risk_expert',
        failure_threshold=3,
        recovery_timeout=30
    )),
}


def with_circuit_breaker(agent_name: str):
    """Decorator to add circuit breaker to a function."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            breaker = circuit_breakers.get(agent_name)
            if not breaker:
                breaker = CircuitBreaker(CircuitBreakerConfig(name=agent_name))
                circuit_breakers[agent_name] = breaker
            
            return await breaker.call(func, *args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            breaker = circuit_breakers.get(agent_name)
            if not breaker:
                breaker = CircuitBreaker(CircuitBreakerConfig(name=agent_name))
                circuit_breakers[agent_name] = breaker
            
            return asyncio.run(breaker.call(func, *args, **kwargs))
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


# ==================== RETRY LOGIC ====================
# Retry logic handles temporary failures by trying again

class RetryPolicy:
    """
    Base class for retry strategies
    
    WHAT IT DOES:
    - Defines how many times to retry failed operations
    - Controls how long to wait between retries
    - Prevents infinite retry loops
    
    COMMON USE CASES:
    - Network timeouts (temporary)
    - API rate limits (recoverable)
    - Database connection issues (transient)
    """
    
    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
        self.max_attempts = max_attempts    # Max retry attempts
        self.base_delay = base_delay        # Starting delay between retries
        self.max_delay = max_delay          # Maximum delay cap


class ExponentialBackoffPolicy(RetryPolicy):
    """
    Smart retry policy that waits longer after each failure
    
    WHAT IT DOES:
    - Starts with short delay, increases exponentially
    - Adds random jitter to prevent synchronized retries
    - Prevents "thundering herd" problem
    
    WHY EXPONENTIAL BACKOFF:
    - If service is struggling, give it more time to recover
    - If many clients retry, jitter spreads them out
    - Reduces load on struggling services
    
    EXAMPLE DELAYS:
    - Attempt 1: Wait 1 second
    - Attempt 2: Wait 2 seconds  
    - Attempt 3: Wait 4 seconds
    - (with jitter: 0.5-1.5s, 1-3s, 2-6s)
    """
    
    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0, 
                 max_delay: float = 60.0, exponential_base: float = 2.0, jitter: bool = True):
        super().__init__(max_attempts, base_delay, max_delay)
        self.exponential_base = exponential_base  # How much to multiply delay each time
        self.jitter = jitter                    # Add randomness to prevent sync
    
    def get_delay(self, attempt: int) -> float:
        """
        Calculate how long to wait before next retry
        
        WHAT IT DOES:
        - Calculates delay using exponential formula
        - Caps delay at maximum to prevent infinite waits
        - Adds jitter if enabled
        
        FORMULA: delay = base_delay * (exponential_base ^ (attempt - 1))
        """
        delay = min(self.base_delay * (self.exponential_base ** (attempt - 1)), self.max_delay)
        
        if self.jitter:
            # Add 50% randomness (multiply by 0.5 to 1.0)
            # This prevents all clients retrying at exactly same time
            delay = delay * (0.5 + random.random() * 0.5)
        
        return delay


# Pre-configured retry policies
DEFAULT_RETRY_POLICY = ExponentialBackoffPolicy(
    max_attempts=3,
    base_delay=1.0,
    max_delay=30.0,
    jitter=True
)

AGGRESSIVE_RETRY_POLICY = ExponentialBackoffPolicy(
    max_attempts=5,
    base_delay=0.5,
    max_delay=15.0,
    jitter=True
)

CONSERVATIVE_RETRY_POLICY = ExponentialBackoffPolicy(
    max_attempts=2,
    base_delay=2.0,
    max_delay=60.0,
    jitter=True
)


def retry_with_policy(policy: RetryPolicy, 
                    retry_on: tuple = (Exception,),
                    retry_on_callback: Optional[Callable] = None):
    """Decorator to add retry logic with custom policy."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(1, policy.max_attempts + 1):
                try:
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)
                    
                    if attempt > 1:
                        logger.info(f"Function {func.__name__} succeeded on attempt {attempt}")
                        metrics.increment_counter('retry_success', 
                                                {'function': func.__name__, 'attempt': str(attempt)})
                    
                    return result
                
                except retry_on as e:
                    last_exception = e
                    logger.warning(f"Function {func.__name__} failed on attempt {attempt}: {str(e)}")
                    
                    if attempt == policy.max_attempts:
                        logger.error(f"Function {func.__name__} failed after {policy.max_attempts} attempts")
                        metrics.increment_counter('retry_exhausted', 
                                                {'function': func.__name__})
                        raise
                    
                    if retry_on_callback:
                        retry_on_callback(e, attempt, policy)
                    
                    delay = policy.get_delay(attempt)
                    logger.info(f"Retrying {func.__name__} in {delay:.2f} seconds...")
                    await asyncio.sleep(delay)
            
            raise last_exception
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(1, policy.max_attempts + 1):
                try:
                    result = func(*args, **kwargs)
                    
                    if attempt > 1:
                        logger.info(f"Function {func.__name__} succeeded on attempt {attempt}")
                        metrics.increment_counter('retry_success', 
                                                {'function': func.__name__, 'attempt': str(attempt)})
                    
                    return result
                
                except retry_on as e:
                    last_exception = e
                    logger.warning(f"Function {func.__name__} failed on attempt {attempt}: {str(e)}")
                    
                    if attempt == policy.max_attempts:
                        logger.error(f"Function {func.__name__} failed after {policy.max_attempts} attempts")
                        metrics.increment_counter('retry_exhausted', 
                                                {'function': func.__name__})
                        raise
                    
                    if retry_on_callback:
                        retry_on_callback(e, attempt, policy)
                    
                    delay = policy.get_delay(attempt)
                    logger.info(f"Retrying {func.__name__} in {delay:.2f} seconds...")
                    time.sleep(delay)
            
            raise last_exception
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


# ==================== TIMEOUT PROTECTION ====================
# Timeout protection prevents functions from running forever

def timeout(seconds: float):
    """
    Decorator to add timeout protection to functions
    
    WHAT IT DOES:
    - Stops function execution after specified time
    - Prevents hanging on slow or stuck operations
    - Works with both async and sync functions
    
    WHY NEEDED:
    - API calls might hang indefinitely
    - Network issues can cause endless waits
    - Prevents system from getting stuck
    
    HOW IT WORKS:
    - Async: Uses asyncio.wait_for() to cancel after timeout
    - Sync: Uses system signals to interrupt after timeout
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                # Wait for function to complete, but cancel after timeout
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                logger.error(f"Function {func.__name__} timed out after {seconds} seconds")
                metrics.increment_counter('timeout', {'function': func.__name__})
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
            
            # Set up alarm signal to interrupt after timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(seconds))
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                signal.alarm(0)  # Cancel the alarm if function completes
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


# ==================== BULKHEAD PATTERN ====================
# Bulkhead pattern limits concurrent executions to prevent overload

class Bulkhead:
    """
    Bulkhead Pattern - Like ship compartments, prevents total failure
    
    WHAT IT DOES:
    - Limits how many functions can run at the same time
    - Prevents system overload from too many concurrent requests
    - Queues excess requests instead of rejecting them
    
    REAL-WORLD ANALOGY:
    Like a restaurant with limited seating - when full, people wait outside
    instead of the restaurant crashing from overcrowding.
    
    WHY NEEDED:
    - Prevents memory exhaustion from too many concurrent operations
    - Controls resource usage (CPU, memory, network)
    - Provides predictable performance under load
    """
    
    def __init__(self, max_concurrent: int = 10):
        self.semaphore = asyncio.Semaphore(max_concurrent)  # Limits concurrent executions
        self.max_concurrent = max_concurrent                # Max allowed at once
        self.current_count = 0                              # Currently running
        self.total_wait_time = 0.0                           # Total time spent waiting
        self.total_requests = 0                              # Total requests processed
    
    async def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function with bulkhead protection
        
        WHAT IT DOES:
        - Waits if too many functions are already running
        - Executes function when slot becomes available
        - Tracks wait times and concurrency metrics
        
        HOW IT WORKS:
        1. Try to acquire semaphore (limited resource)
        2. If not available, wait until it is
        3. Execute function once acquired
        4. Release semaphore when done
        
        EXAMPLE:
        If max_concurrent=5 and 10 requests come in:
        - First 5 execute immediately
        - Next 5 wait until one of the first 5 finishes
        """
        start_time = time.time()
        self.total_requests += 1
        
        # Wait for available slot (this is the "bulkhead" part)
        async with self.semaphore:
            wait_time = time.time() - start_time
            self.total_wait_time += wait_time
            
            if wait_time > 0:
                logger.debug(f"Bulkhead waited {wait_time:.2f} seconds for semaphore")
                metrics.histogram('bulkhead_wait_time', wait_time, {'max_concurrent': str(self.max_concurrent)})
            
            try:
                self.current_count += 1
                metrics.set_gauge('bulkhead_current', self.current_count, {'max_concurrent': str(self.max_concurrent)})
                
                # Execute the actual function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                return result
            finally:
                self.current_count -= 1
                metrics.set_gauge('bulkhead_current', self.current_count, {'max_concurrent': str(self.max_concurrent)})
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bulkhead statistics."""
        avg_wait = self.total_wait_time / self.total_requests if self.total_requests > 0 else 0
        return {
            'max_concurrent': self.max_concurrent,
            'current_count': self.current_count,
            'total_requests': self.total_requests,
            'average_wait_time': avg_wait
        }


# Pre-configured bulkheads for different operations
bulkheads = {
    'agent_calls': Bulkhead(max_concurrent=5),
    'workflow_execution': Bulkhead(max_concurrent=10),
    'api_requests': Bulkhead(max_concurrent=20),
}


def with_bulkhead(bulkhead_name: str):
    """Decorator to add bulkhead protection to a function."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            bulkhead = bulkheads.get(bulkhead_name)
            if not bulkhead:
                bulkhead = Bulkhead()
                bulkheads[bulkhead_name] = bulkhead
            
            return await bulkhead.execute(func, *args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            bulkhead = bulkheads.get(bulkhead_name)
            if not bulkhead:
                bulkhead = Bulkhead()
                bulkheads[bulkhead_name] = bulkhead
            
            return asyncio.run(bulkhead.execute(func, *args, **kwargs))
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


# ==================== COMBINED RESILIENCE DECORATOR ====================
# One decorator to apply all protection patterns

def resilient(agent_name: Optional[str] = None,
             retry_policy: Optional[RetryPolicy] = None,
             timeout_seconds: Optional[float] = None,
             bulkhead_name: Optional[str] = None):
    """
    All-in-one decorator that applies multiple resilience patterns
    
    WHAT IT DOES:
    - Combines circuit breaker, retry, timeout, and bulkhead protection
    - Applies protections in the right order for maximum effectiveness
    - Provides simple way to protect any function
    
    ORDER OF OPERATIONS (important!):
    1. Bulkhead: Limit concurrent executions first
    2. Timeout: Prevent hanging during execution
    3. Retry: Try again on temporary failures
    4. Circuit Breaker: Stop calling if consistently failing
    
    WHY THIS ORDER:
    - Bulkhead first: Prevents system overload
    - Timeout second: Prevents individual calls from hanging
    - Retry third: Handles temporary issues
    - Circuit breaker last: Prevents cascading failures
    
    USAGE EXAMPLE:
    @resilient(
        agent_name="equity_expert",
        retry_policy=DEFAULT_RETRY_POLICY,
        timeout_seconds=30,
        bulkhead_name="agent_calls"
    )
    async def analyze_stock(symbol):
        # This function is now fully protected!
        pass
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # Apply decorators in specific order for best protection
        
        if bulkhead_name:
            func = with_bulkhead(bulkhead_name)(func)      # 1. Limit concurrency
        
        if timeout_seconds:
            func = timeout(timeout_seconds)(func)          # 2. Prevent hanging
        
        if retry_policy:
            func = retry_with_policy(retry_policy)(func)   # 3. Retry on failure
        
        if agent_name:
            func = with_circuit_breaker(agent_name)(func)   # 4. Stop if failing
        
        return func
    return decorator


# ==================== MONITORING FUNCTIONS ====================
# Functions to check system health and status

def get_circuit_breaker_status() -> Dict[str, Any]:
    """
    Get status of all circuit breakers for health monitoring
    
    WHAT IT DOES:
    - Returns current state of every circuit breaker
    - Used for health checks and monitoring dashboards
    - Helps identify which agents are having problems
    
    RETURNS:
    Dictionary with circuit breaker names and their current status
    """
    return {name: breaker.get_state() for name, breaker in circuit_breakers.items()}


def get_bulkhead_stats() -> Dict[str, Any]:
    """
    Get statistics of all bulkheads for performance monitoring
    
    WHAT IT DOES:
    - Returns usage statistics for all bulkheads
    - Shows how often requests are waiting
    - Used for performance tuning and monitoring
    
    RETURNS:
    Dictionary with bulkhead names and their performance stats
    """
    return {name: bulkhead.get_stats() for name, bulkhead in bulkheads.items()}


def reset_all_circuit_breakers():
    """
    Manually reset all circuit breakers to working state
    
    WHAT IT DOES:
    - Forces all circuit breakers back to CLOSED state
    - Clears all failure histories
    - Used for manual recovery after issues are resolved
    
    WHEN TO USE:
    - After fixing underlying problems
    - During maintenance or testing
    - When you want to give agents a fresh start
    """
    for name, breaker in circuit_breakers.items():
        breaker.reset()
        logger.info(f"Reset circuit breaker: {name}")
