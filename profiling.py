import time
import functools


def timing(name=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            label = name if name else func.__name__
            print(f"time {label}: {(end-start)*1000:.2f} ms")
            return result
        return wrapper
    return decorator
