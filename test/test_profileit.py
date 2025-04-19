"""
Source https://github.com/pytorch/tutorials/blob/main/beginner_source/profiler.py

"""
import torch
import torch.nn as nn
from profileit import profileit, ScheduleArgs


class SimpleModel(nn.Module):
    def __init__(self, input_dim=10, output_dim=1, bias=True):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias=bias)

    def linear_pass(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    def mask_pass(self, x: torch.Tensor, mask):
        threshold = x.sum(1).mean()
        return (mask > threshold).nonzero(as_tuple=True)

    def forward(self, x, mask):
        x = self.linear_pass(x)
        return x, self.mask_pass(x, mask)


if __name__ == "__main__":

    import sys
    import os
    device = None
    if len(sys.argv) > 1:
        device = sys.argv[1]

    model = SimpleModel(256, 10)
    
    if device:
        model = model.to(device)

    with profileit(
        model,
        schedule=ScheduleArgs(warmup=1, active=2),
        trace_report_dir="trace_report",
        seed=42,
        profile_memory=True,
        #with_stack=True,
    ) as (profiled_model, step_generator):
        input = torch.rand((128, 256), device=device)
        mask = torch.rand((256, 256, 256), dtype=torch.float, device=device)
        for step in step_generator:
            print(f"Step {step}")
            out, idx = profiled_model(input, mask)
        
    # Output
    """
Step 0
Step 1
Step 2
==== SimpleModel Results (CPU) ====
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                         Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem    # of Calls  
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                aten::nonzero        94.48%     117.766ms        96.48%     120.258ms      60.129ms           0 b           0 b             2  
    aten::_local_scalar_dense         1.55%       1.931ms         1.57%       1.961ms     980.287us           0 b         -16 b             2  
        SimpleModel.mask_pass         0.84%       1.048ms        98.04%     122.200ms      61.100ms           0 b           0 b             2  
                 aten::linear         0.70%     870.704us         0.74%     928.037us     464.018us           0 b           0 b             2  
                ProfilerStep*         0.53%     655.498us       100.00%     124.647ms      62.324ms           0 b           0 b             2  
          aten::count_nonzero         0.39%     491.956us         0.41%     506.997us     253.499us           0 b           0 b             2  
                       Linear         0.32%     392.830us         1.06%       1.321ms     660.434us           0 b           0 b             2  
                  SimpleModel         0.29%     361.667us        99.47%     123.992ms      61.996ms           0 b           0 b             2  
                    aten::sum         0.23%     286.416us         0.23%     286.416us     143.208us           0 b           0 b             2  
                     aten::gt         0.16%     195.374us         0.16%     195.374us      97.687us           0 b           0 b             2  
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 124.647ms
Profiler results saved to trace_report/SimpleModel_trace_3.1744986809924269000.json
"""
