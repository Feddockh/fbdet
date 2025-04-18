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
    transforms = v2.Compose([
        v2.Resize((1024, 1024), antialias=True), # Higher for finer details
        v2.RandomHorizontalFlip(p=0.5),
    ])

    # Create dataset and dataloader
    dataset = MultiCamDataset(
        base_dir=DATA_DIR,
        cameras=cameras,
        set_type=SetType.TRAIN,
        transforms=transforms
    )
    class_names = dataset.get_class_names()
    dataloader = MultiCamDataloader(
        dataset,
        batch_size=5,
        shuffle=True,
        num_workers=4
    )

    # Initialize model and trainer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiCamModel(cameras=camera_names, num_classes=len(class_names)+1, pretrained=True)
    trainer = MultiCamTrainer(
        model=model,
        dataloader=dataloader,
        device=device
    )

    # Begin training
    print(f"Starting training for {10} epochs on device {device}")
    trainer.fit(num_epochs=10)


if __name__ == '__main__':
    main()