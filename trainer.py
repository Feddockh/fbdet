import os
import torch
from torch.optim import SGD, Optimizer
from loader import MultiCamDataloader
from model import MultiCamModel
from typing import List, Dict, Optional


class MultiCamTrainer:
    def __init__(self, model: MultiCamModel, dataloader: MultiCamDataloader,
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 optimizer: Optional[Optimizer] = None, lr: float = 1e-3, momentum: float = 0.9, \
                 weight_decay: float = 5e-4, checkpoint_dir: str = "checkpoints"):
        """
        Trainer for MultiCamModel using a MultiCamDataLoader.

        Args:
            model: MultiCamModel instance
            dataloader: DataLoader yielding batches as dict[str, List[(img, target)]]
            device: device to train on
            lr: learning rate for optimizer
            momentum: momentum for SGD
            weight_decay: weight decay for SGD
        """
        self.device = device
        self.model = model.to(self.device)
        self.dataloader = dataloader
        self.optimizer = optimizer or SGD(
            self.model.parameters(), lr=lr,
            momentum=momentum, weight_decay=weight_decay
        )
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def loss(self, inputs: Dict[str, List[torch.Tensor]],
             targets: Dict[str, List[Dict]]) -> torch.Tensor:
        losses = self.model(inputs, targets)
        return sum(v for v in losses.values())

    def train_epoch(self, epoch_idx: int, print_freq: int = 10) -> float:
        self.model.train()
        running_loss = 0.0
        count = 0
        for batch_idx, batch in enumerate(self.dataloader):
            inputs, targets = {}, {}
            for cam, sample in batch.items():
                imgs, targs = zip(*sample)
                inputs[cam] = [img.to(self.device) for img in imgs]
                targets[cam] = [{k: v.to(self.device) for k, v in t.items()} for t in targs]

            self.optimizer.zero_grad()
            loss = self.loss(inputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            count += 1
            if batch_idx % print_freq == 0:
                print(f"Epoch [{epoch_idx}] Batch [{batch_idx}/{len(self.dataloader)}] Loss: {loss.item():.4f}")

        avg_loss = running_loss / count if count else 0.0
        print(f"Epoch [{epoch_idx}] Average Loss: {avg_loss:.4f}")
        ckpt = os.path.join(self.checkpoint_dir, f"epoch_{epoch_idx}.pth")
        if hasattr(self.model, 'save_checkpoint'):
            self.model.save_checkpoint(ckpt)
            print(f"Saved checkpoint: {ckpt}")
        return avg_loss

    def fit(self, num_epochs: int, start_epoch: int = 1):
        for e in range(start_epoch, start_epoch + num_epochs):
            self.train_epoch(e)