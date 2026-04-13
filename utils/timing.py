import time
from functools import wraps

def timing(name: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            print(f"time {name}: {(end - start) * 1000:.2f} ms")
            return result
        return wrapper
    return decorator
