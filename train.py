import os
import torch
import torchvision.transforms.v2 as v2
from camera import Camera
from dataset import MultiCamDataset, SetType
from loader import MultiCamDataloader
from model import MultiCamModel
from trainer import MultiCamTrainer


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DATA_DIR = os.path.join(BASE_DIR, "rivendale_dataset")
DATA_DIR = os.path.join(BASE_DIR, "erwiam_dataset")

def main():
    # Prepare camera list
    camera_names = ['cam0']
    cameras = [Camera(name) for name in camera_names]

    # Define transforms
    train_transforms = v2.Compose([
        v2.RandomResizedCrop((1024, 1024), antialias=True),       # random crop + scale
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(0.2, 0.2, 0.2, 0.1),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transforms = v2.Compose([
        v2.Resize((1024, 1024), antialias=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create dataset and dataloader for the training set
    dataset = MultiCamDataset(
        base_dir=DATA_DIR,
        cameras=cameras,
        set_type=SetType.TRAIN,
        transforms=train_transforms
    )
    class_names = dataset.get_class_names()
    dataloader = MultiCamDataloader(
        dataset,
        batch_size=5,
        shuffle=True,
        num_workers=4
    )

    # Create a dataset and dataloader for the validation set
    val_dataset = MultiCamDataset(
        base_dir=DATA_DIR,
        cameras=cameras,
        set_type=SetType.VAL,
        transforms=val_transforms
    )
    val_dataloader = MultiCamDataloader(
        val_dataset,
        batch_size=5,
        shuffle=False,
        num_workers=4
    )

    # Initialize model and trainer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiCamModel(cameras=camera_names, num_classes=len(class_names)+1, pretrained=True)
    trainer = MultiCamTrainer(
        model=model,
        dataloader=dataloader,
        val_dataloader=val_dataloader,
        device=device
    )

    # Begin training
    num_epochs = 40
    print(f"Starting training for {num_epochs} epochs on device {device}")
    trainer.fit(num_epochs)


if __name__ == '__main__':
    main()