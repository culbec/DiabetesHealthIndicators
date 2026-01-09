import time


def time_func(func, num_decimals: int = 8):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start_time

        # Automatically use self.logger.info if available
        self_ = args[0] if args else None
        if self_ is not None and hasattr(self_, "logger") and hasattr(self_.logger, "debug"):
            reporter = self_.logger.debug
        elif kwargs.get("logger") and hasattr(kwargs["logger"], "info"):
            reporter = kwargs["logger"].debug
        else:
            reporter = print
        reporter(f"Function '{func.__name__}' executed in {elapsed:.{num_decimals}f} seconds.")
        return result

    return wrapper
