import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

# Basic in-memory storage for early observability (Phase 2)
# Phase 10 will upgrade this to Prometheus/Grafana
metrics = defaultdict(int)
latencies = defaultdict(list)

class ObservabilityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            # Record status codes
            metrics[f"status_{response.status_code}"] += 1
            
            # Record endpoint latency
            process_time = time.time() - start_time
            # Keep only last 100 entries per path to avoid memory leak
            if len(latencies[request.url.path]) >= 100:
                latencies[request.url.path].pop(0)
            latencies[request.url.path].append(process_time)
            
            # Add custom header for immediate feedback
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
        except Exception as e:
            metrics["failures"] += 1
            logger.error(f"Request failed: {str(e)}")
            raise e
