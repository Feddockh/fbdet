import os
import torch
import matplotlib
matplotlib.use('Agg') # Avoids "DataLoader worker exited unexpectedly" error
from torch.optim import SGD, Optimizer
from loader import MultiCamDataloader
from model import MultiCamModel
from typing import List, Dict, Optional
from utils.visual import plot_loss
from tqdm.auto import tqdm


class MultiCamTrainer:
    def __init__(self, model: MultiCamModel, dataloader: MultiCamDataloader,
                 val_dataloader: Optional[MultiCamDataloader] = None,
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 optimizer: Optional[Optimizer] = None, lr: float = 1e-3, momentum: float = 0.9, \
                 weight_decay: float = 5e-4, checkpoint_dir: str = "checkpoints"):
        """
        Trainer for MultiCamModel using a MultiCamDataLoader.

        Args:
            model: MultiCamModel instance
            dataloader: DataLoader yielding batches as dict[str, List[(img, target)]]
            val_dataloader: Optional DataLoader for validation
            device: Device to train on
            optimizer: Optimizer for training (default: SGD)
            lr: Learning rate for optimizer
            momentum: Momentum for SGD
            weight_decay: Weight decay for SGD
            checkpoint_dir: Directory to save checkpoints
        """
        self.device = device
        self.model = model.to(self.device)
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer or SGD(
            self.model.parameters(), lr=lr,
            momentum=momentum, weight_decay=weight_decay
        )
        if self.val_dataloader is not None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=self.optimizer,
                mode='min',
                factor=0.1,
                patience=3,
                verbose=True
            )
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer=self.optimizer,
                step_size=5,
                gamma=0.1
            )
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Loss tracking
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, epoch_idx: int, save_path: str = None) -> float:
        """
        Train model for one epoch and compute loss.
        """
        self.model.train()

        # Iterate over each batch
        total_loss = 0.0
        num_batches = len(self.dataloader)
        for batch in tqdm(self.dataloader, total=num_batches, desc=f"Epoch [{epoch_idx}]", unit="batch"):

            # Move the images and targets to the device
            inputs, targets = {}, {}
            for cam, sample in batch.items():
                imgs, targs = zip(*sample)
                inputs[cam] = [img.to(self.device) for img in imgs]
                targets[cam] = [{k: v.to(self.device) for k, v in t.items()} for t in targs]

            # Forward pass (gradient update)
            self.optimizer.zero_grad()
            losses = self.model(inputs, targets) # losses contains loss_classifier, loss_box_reg, loss_objectness, loss_rpn_box_reg
            loss = sum(v for v in losses.values())
            loss.backward()
            self.optimizer.step()

            # Update running loss
            total_loss += loss.item()

        # Compute average loss for the epoch
        avg_loss = total_loss / num_batches
        print(f"Epoch [{epoch_idx}] Average Training Loss: {avg_loss:.4f}")

        return avg_loss
    
    def validate_epoch(self, epoch_idx: int) -> float:
        """
        Compute validation loss over the validation dataset.
        """
        running_loss = 0.0
        num_batches = len(self.val_dataloader)
        with torch.no_grad():
            for batch in self.val_dataloader:
                inputs, targets = {}, {}
                for cam, sample in batch.items():
                    imgs, targs = zip(*sample)
                    inputs[cam] = [img.to(self.device) for img in imgs]
                    targets[cam] = [{k: v.to(self.device) for k, v in t.items()} for t in targs]

                # Forward pass (no gradient update)
                losses = self.model(inputs, targets)
                loss = sum(v for v in losses.values())
                running_loss += loss.item()

        avg_loss = running_loss / num_batches
        print(f"Epoch [{epoch_idx}] Average Validation Loss: {avg_loss:.4f}")
        return avg_loss

    def fit(self, num_epochs: int, start_epoch: int = 1, save_checkpoint: bool = True):
        """
        Run training (and validation if given loader) for N epochs, plotting losses after each epoch.
        """
        for epoch in range(start_epoch, start_epoch + num_epochs):

            # Train for an epoch and compute loss
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)

            # Save the model checkpoint
            if save_checkpoint:
                checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
                self.model.save_checkpoint(checkpoint_path)

            # Compute loss on validation set if loader provided
            if self.val_dataloader is not None:
                val_loss = self.validate_epoch(epoch)
                self.val_losses.append(val_loss)

            # Step the learning rate scheduler
            if self.val_dataloader is not None:
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            # Print out the current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch}] Learning Rate: {current_lr:.2e}")

            # Plot and save loss curves
            plot_path = os.path.join(self.checkpoint_dir, "loss_plot.png")
            if self.val_dataloader is not None:
                plot_loss(self.train_losses, self.val_losses, save_path=plot_path)
            else:
                plot_loss(self.train_losses, save_path=plot_path)
            # print(f"Saved loss plot: {plot_path}")