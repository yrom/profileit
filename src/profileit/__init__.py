import warnings
from contextlib import contextmanager
from types import MethodType
from typing import Any, Callable, Generator, Optional, Tuple, TypeVar, Union

import numpy as np
import torch
from torch.profiler import profile, supported_activities
from torch.profiler import schedule as _schedule

from ._torch_profile import profile_inject, profile_trace_handler, sync

M = TypeVar("M")


class ScheduleArgs:
    def __init__(self, wait: int = 0, warmup: int = 0, active: int = 1):
        self.wait = wait
        self.warmup = warmup
        self.active = active
        self.num_steps = self.wait + self.warmup + self.active


@contextmanager
def profileit(
    *models: M,
    ignore_fn: Optional[Callable[[str, Union[MethodType, Any]], bool]] = None,
    schedule=ScheduleArgs(),
    trace_report_dir: Optional[str] = None,
    seed: Optional[int] = None,
    **profile_args,
):
    """A context manager for profiling PyTorch model inference.

    This function wraps a PyTorch model with profiling capabilities, allowing detailed
    performance analysis of model inference. It uses PyTorch's profiler to track
    operations, memory usage, and other performance metrics.

    Args:
        models (Tuple[M, ...]): The model(s) to be profiled. Can be a single model or a list of models.
        ignore_fn (Optional[Callable[[str, Union[MethodType, Any], bool]]): A function to filter
            which operations should not be profiled. Returns true if the operation should be ignored in profiling.
        trace_report_dir: (Optional[str]): Directory to save the profiling report. If None, no report is saved.
        schedule (ScheduleArgs): Schedule for profiling steps. It defines the wait, warmup,
            active, and repeat steps for profiling. see torch.profiler.schedule for details.
        seed (Optional[int]): Random seed for reproducibility. If None, no seed is set.
        **profile_args: Additional arguments to pass to torch.profiler.profile:
            - activities (Optional[set]): Set of activities to profile (e.g.,
                {ProfilerActivity.CPU, ProfilerActivity.CUDA}).
            - profile_memory (bool): Whether to profile memory usage (default: False)
            - with_stack (bool): Whether to record stack traces (default: False)

    Yields:
        tuple: A tuple containing:
            - The instrumented models ready for profiling
            - A generator will yield profiling steps until the profiling is done.
    
    Example:
        >>> with profileit(model, schedule=ScheduleArgs(warmup=1, active=1)) as (profiled_model, step_generator):
        >>>     for step in step_generator:
        >>>         # Run your model inference here
        >>>         print(f"Step {step}")
        >>>         out, idx = profiled_model(input, mask)
    """
    model_name = models[0].__class__.__name__
    for i in range(len(models)):
        profile_inject(models[i], ignore_fn=ignore_fn)
    
    profile_schedule = _schedule(
        wait=schedule.wait,
        warmup=schedule.warmup,
        active=schedule.active,
        repeat=0,
    )
    on_trace_ready = profile_trace_handler(trace_report_dir, model_name=model_name)
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    prof = profile(
        activities=profile_args.get("activities", supported_activities()),
        profile_memory=profile_args.get("profile_memory", False),
        with_stack=profile_args.get("with_stack", False),
        schedule=profile_schedule,
        on_trace_ready=on_trace_ready,
    )
    def step_generator():
        for i in range(schedule.num_steps):
            yield i
            prof.step()
    prof.__enter__()
    try:
        yield *models, step_generator()
    except Exception as e:
        warnings.warn(
            "Exception occurred during profiling",
            category=RuntimeWarning,
            source=e
        )
        raise
    finally:
        sync()
        prof.__exit__(None, None, None)


__all__ = ["profileit", "ScheduleArgs"]
