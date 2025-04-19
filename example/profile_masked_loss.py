"""
Follow this example to profile a masked loss function in PyTorch.
This example is based on the article
https://medium.com/data-science/pytorch-model-performance-analysis-and-optimization-part-3-1c5876d78fe2

Run version 1, 2, and 3 of the masked loss function to compare the performance.
```sh
python example/profile_masked_loss.py "cuda" "1"
python example/profile_masked_loss.py "cuda" "2"
python example/profile_masked_loss.py "cuda" "3"
```
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


from profileit import ScheduleArgs, profileit


class Net(nn.Module):
    def __init__(self, num_hidden=10, num_classes=10):
        super().__init__()
        self.conv_in = nn.Conv2d(3, 10, 3, padding="same")
        hidden = []
        for i in range(num_hidden):
            hidden.append(nn.Conv2d(10, 10, 3, padding="same"))
            hidden.append(nn.ReLU())

        self.hidden = nn.Sequential(*hidden)
        self.conv_out = nn.Conv2d(10, num_classes, 3, padding="same")

    def forward(self, x):
        x = F.relu(self.conv_in(x))
        x = self.hidden(x)
        x = self.conv_out(x)
        return x


class MaskedLossV1(nn.Module):
    def __init__(self, ignore_val=-1, num_classes=10):
        super().__init__()
        self.ignore_val = ignore_val
        self.num_classes = num_classes
        self.loss = torch.nn.CrossEntropyLoss()

    def cross_entropy(self, pred: Tensor, target: Tensor) -> Tensor:
        # create a boolean mask of valid labels
        mask = target != self.ignore_val
        # permute the logits in preparation for masking
        permuted_pred = torch.permute(pred, [0, 2, 3, 1])

        # apply the boolean mask to the targets and logits
        masked_target = target[mask]
        masked_pred = permuted_pred[mask.unsqueeze(-1).expand(-1, -1, -1, self.num_classes)]
        masked_pred = masked_pred.reshape(-1, self.num_classes)

        # calculate the cross-entropy loss
        loss = self.loss(masked_pred, masked_target)
        return loss

    def ignore_background(self, target: Tensor) -> Tensor:
        # discover all indices where target label is "background"
        inds = torch.nonzero(target == self.num_classes - 1, as_tuple=True)

        # reset all "background" labels to the ignore index
        target[inds] = self.ignore_val
        return target

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        # ignore background labels
        target = self.ignore_background(target)

        # retrieve a list of unique elements in target
        unique = torch.unique(target)

        # check if the number of unique items pass the threshold
        ignore_loss = torch.numel(unique) < 2

        # calculate the cross-entropy loss
        loss = self.cross_entropy(pred, target)

        # zero the loss in the case that the number of unique elements
        # is below the threshold
        if ignore_loss:
            loss = 0.0 * loss

        return loss


class MaskedLossV2(MaskedLossV1):
    def ignore_background(self, target: Tensor) -> Tensor:
        # instead torch.nonzero, use torch.where
        # torch.nonzero causes gpu device sync
        return torch.where(target == self.num_classes - 1, self.ignore_val * torch.ones_like(target), target)


class MaskedLossV3(MaskedLossV1):
    def __init__(self, ignore_val=-1, num_classes=10):
        super().__init__(ignore_val, num_classes)
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return self.loss(pred, target)


# A dataset with random images and label maps
class FakeDataset(Dataset):
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = [256, 256]

    def __len__(self):
        return 1000

    def __getitem__(self, index):
        rand_image = torch.randn([3] + self.img_size, dtype=torch.float32)
        rand_label = torch.randint(low=-1, high=self.num_classes, size=self.img_size)
        return rand_image, rand_label


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        device = sys.argv[1]
        if len(sys.argv) > 2:
            version = sys.argv[2]
    else:
        device = None
        version = "1"

    num_classes = 5
    train_set = FakeDataset(num_classes=num_classes)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    model = Net(num_classes=num_classes).to(device)
    if version == "1":
        print("Using MaskedLossV1")
        criterion = MaskedLossV1(num_classes=num_classes).to(device)
    elif version == "2":
        print("Using MaskedLossV2")
        criterion = MaskedLossV2(num_classes=num_classes).to(device)
    elif version == "3":
        print("Using MaskedLossV3")
        criterion = MaskedLossV3(num_classes=num_classes).to(device)
    else:
        raise ValueError(f"Unknown version: {version}")

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()

    # training loop wrapped with profileit
    with profileit(
        model,
        criterion,
        schedule=ScheduleArgs(wait=0, warmup=1, active=3),
        trace_report_dir="trace_report",
        seed=420,
        profile_memory=True,
        # record_shapes=True,
        # with_stack=True,
    ) as (profiled_model, profiled_criterion, step_generator):
        loader = iter(train_loader)
        for step in step_generator:
            print(f"Step {step}")
            (inputs, labels) = next(loader)
            if device:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

            outputs = profiled_model(inputs)
            loss = profiled_criterion(outputs, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        print(f"Loss: {loss.item()}")
