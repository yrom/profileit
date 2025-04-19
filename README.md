# ProfileIt

ProfileIt is a Python package designed to simplify profiling of PyTorch models. Based on [`torch.profiler`](https://pytorch.org/tutorials/beginner/profiler.html).


## Installation
Install the package using pip:

```bash
pip install profileit
```

## Usage
Here is a basic example of how to use ProfileIt with a PyTorch model:

```python

from profileit import profileit,ScheduleArgs
# Profile the model
with profileit(
    model,
    schedule=ScheduleArgs(warmup=1, active=2),
    trace_report_dir="traces",
    seed=42,
    profile_memory=True,
    with_stack=True,
) as (profiled_model, step_generator):

    for step in step_generator:
        print(f"Step {step}")
        out, idx = profiled_model(input, mask)
```

## License
MIT