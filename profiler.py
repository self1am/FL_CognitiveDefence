#!/usr/bin/env python3
"""
Profile where time is spent in a Flower round
"""
import logging
import time
from functools import wraps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Patch this into your training/eval functions
_section_times = {}

def profile_section(section_name):
    """Decorator to time code sections"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            logger.info(f"[PROFILE] Starting: {section_name}")
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start
                _section_times[section_name] = _section_times.get(section_name, 0) + elapsed
                logger.info(f"[PROFILE] Completed: {section_name} ({elapsed:.2f}s)")
                return result
            except Exception as e:
                elapsed = time.time() - start
                logger.error(f"[PROFILE] FAILED: {section_name} after {elapsed:.2f}s - {e}")
                raise
        return wrapper
    return decorator

# Add to your strategy or client code:
# @profile_section("client_training")
# def train(...):
#     ...

# @profile_section("aggregation")
# def aggregate_fit(...):
#     ...

# Then at end of round:
logger.info(f"[PROFILE] ROUND TIMES: {_section_times}")
