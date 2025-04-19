import os
from types import BuiltinMethodType, MethodType, MethodWrapperType
from typing import Callable, Optional, TypeVar, Union

import torch
import wrapt
import inspect

import torch.nn as nn
from torch.profiler import record_function


def sync():
    """
    synchronize the GPU or MPS device if available.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.mps.is_available():
        torch.mps.synchronize()


class BoundCallableWrapper(wrapt.ObjectProxy):
    def __init__(self, wrapped, wrapper):
        super(BoundCallableWrapper, self).__init__(wrapped)

        # print(f"Wrapping callable: {wrapped.__qualname__}")
        self._self_wrapper = wrapper
        self._self_enabled = True

    def __get__(self, instance, owner):
        return self

    def __call__(self, *args, **kwargs):
        if not self._self_enabled:
            return self.__wrapped__(*args, **kwargs)
        return self._self_wrapper(self.__wrapped__, args, kwargs)


class ObjectWrapper(wrapt.ObjectProxy):
    def __init__(self, wrapped, ignore_fn: Optional[Callable[[str, MethodType], bool]] = None):
        if isinstance(wrapped, wrapt.ObjectProxy):
            raise TypeError("Cannot wrap an already wrapped object")
        super(ObjectWrapper, self).__init__(wrapped)

        self._self_wrapper = wrapped

        def _self_method_wrapper(wrapped_m, args, kwargs):
            with record_function(f"{self._self_wrapper.__class__.__name__}.{wrapped_m.__name__}"):
                return wrapped_m(*args, **kwargs)

        _class_methods = [
            (n, m)
            for (n, m) in wrapped.__dict__.items()
            if inspect.ismethod(m)
            and not isinstance(m, BuiltinMethodType)
            and not isinstance(m, MethodWrapperType)
            and not n.startswith("_")
            or (ignore_fn is not None and ignore_fn(n, m))
            # and not n.endswith("_")
        ]

        for name, method in _class_methods:
            if ignore_fn is not None and ignore_fn(name, method):
                continue
            if isinstance(method, wrapt.ObjectProxy):
                # print(f"Skipping already wrapped method: {name} in {wrapped.__class__.__name__}")
                continue
            # Wrap the method with a BoundCallableWrapper
            #print(f"Wrapping object method: {wrapped.__class__.__name__}.{method.__name__}")
            wrapped_method = BoundCallableWrapper(method, _self_method_wrapper)
            setattr(wrapped, name, wrapped_method)
        
        # Wrap the sub modules when the wrapped is not a nn.Module
        if not isinstance(wrapped, nn.Module):
            child_modules = [
                (n, m)
                for (n, m) in wrapped.__dict__.items()
                if isinstance(m, nn.Module) and not isinstance(m, wrapt.ObjectProxy)
            ]
            for n, module in child_modules:
                # Wrap the module with ModuleWrapper
                print(f"Wrapping module: {wrapped.__class__.__name__}.{n} for {module.__class__.__name__}")
                wrapped_module = ModuleWrapper(module, ignore_fn=ignore_fn)
                setattr(wrapped, n, wrapped_module)


class ModuleWrapper(ObjectWrapper):
    def __init__(self, wrapped: nn.Module, ignore_fn: Optional[Callable[[str, MethodType], bool]] = None, skip_children=False):
        if isinstance(wrapped, wrapt.ObjectProxy):
            raise TypeError("Cannot wrap an already wrapped object")
        if not isinstance(wrapped, nn.Module):
            raise TypeError("ModuleWrapper can only wrap nn.Module objects")

        super(ModuleWrapper, self).__init__(
            wrapped,
            ignore_fn=lambda n, m: n in ["forward", "to", "extra_repr"] or ignore_fn and ignore_fn(n, m),
        )
        #print(f"Wrapped module: {wrapped.__class__.__name__}")
        # Module attributes
        if not skip_children:
            child_modules = [
                (n, c)
                for n, c in wrapped.named_modules()
                if not isinstance(c, wrapt.ObjectProxy)
                and len(n) > 0
                and not n.startswith("_")
            ]
            
            for n, child in child_modules:
                # Wrap the child module with ModuleWrapper
                print(f"Wrapping child module {n} of {wrapped.__class__.__name__}: {child.__class__.__name__}")
                wrapped_child = ModuleWrapper(child, ignore_fn=ignore_fn, skip_children=True)
                # replace the child module with the wrapped one
                setattr(wrapped, n, wrapped_child)


        def _self_call_wrapper(wrapped_m, args, kwargs):
            with record_function(f"{self._self_wrapper.__class__.__name__}"):
                return wrapped_m(*args, **kwargs)

        wrapped._call_impl = BoundCallableWrapper(
            wrapped._call_impl,
            _self_call_wrapper,
        )

    def __call__(self, *args, **kwargs):
        name = self.__wrapped__.__class__.__qualname__
        bound_call = (
            self.__wrapped__._call_impl
            if hasattr(self.__wrapped__, "_call_impl") and isinstance(self.__wrapped__._call_impl, BoundCallableWrapper)
            else None
        )
        try:
            if bound_call:
                bound_call._self_enabled = False
            with record_function(name):
                return self.__wrapped__(*args, **kwargs)
        finally:
            if bound_call:
                bound_call._self_enabled = True


T = TypeVar("T")


def profile_inject(obj: T, ignore_fn: Optional[Callable[[str, MethodType], bool]] = None) -> T:
    if isinstance(obj, wrapt.ObjectProxy):
        return obj
    if inspect.isbuiltin(obj) or inspect.isclass(obj):
        return obj
    if isinstance(obj, nn.Module):
        return profile_module(obj, ignore_fn=ignore_fn)
    if isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = profile_inject(obj[i], ignore_fn=ignore_fn)
        return obj
    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = profile_inject(v, ignore_fn=ignore_fn)
        return obj
    if isinstance(obj, tuple):
        obj = tuple(profile_inject(item, ignore_fn=ignore_fn) for item in obj)
        return obj
    if inspect.isfunction(obj):

        def f_wrapper(wrapped_m, args, kwargs):
            with record_function(f"{obj.__name__}"):
                return wrapped_m(*args, **kwargs)

        return BoundCallableWrapper(obj, f_wrapper)
    return ObjectWrapper(obj, ignore_fn=ignore_fn)


def profile_module(module: nn.Module, ignore_fn=None) -> nn.Module:
    return ModuleWrapper(module, ignore_fn=ignore_fn)


def profile_trace_handler(
    dir: Union[None, str, os.PathLike], model_name: Optional[str] = None
) -> Callable[[torch.profiler.profile], None]:
    from torch.profiler import ProfilerActivity, profile
    import time

    if model_name is None:
        model_name = ""

    def _trace_handler(p: profile):
        group_by_stack_n = 5  if p.with_stack else 0
        if ProfilerActivity.CPU in p.activities:
            print("====", model_name, "Results (CPU)", "====")
            print(p.key_averages(group_by_stack_n=group_by_stack_n).table(sort_by="self_cpu_time_total", row_limit=10))
        if ProfilerActivity.CUDA in p.activities and torch.cuda.is_available():
            print("====", model_name, "Results (CUDA)", "====")
            print(p.key_averages(group_by_stack_n=group_by_stack_n).table(sort_by="self_cuda_time_total", row_limit=10))
        if dir is None:
            return
        if not os.path.isdir(dir):
            try:
                os.makedirs(dir, exist_ok=True)
            except Exception as e:
                print(f"Failed to create directory {dir}: {e}")
                return

        file_name = f"{model_name}_trace_step{p.step_num}_{int(time.time())}.json"
        trace_file = os.path.join(dir, file_name)
        p.export_chrome_trace(trace_file)
        print(f"Profiler results saved to {trace_file}")

    return _trace_handler
