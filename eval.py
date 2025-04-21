import os
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat
from tqdm.auto import tqdm

from dataset import MultiCamDataset, SetType
from loader import MultiCamDataloader
from model import MultiCamModel
from camera import Camera
from utils.visual import plot


# Parameters
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(BASE_DIR, "erwiam_dataset")
SETTYPE       = SetType.VAL
CAMERAS       = ['cam0']
CHECKPOINT    = os.path.join(BASE_DIR, "checkpoints", "epoch_10.pth")
CLASS_NAMES   = ['bg', 'flower', 'shoot', 'maybe', 'leaf']
NUM_CLASSES   = 5 # including background
SAMPLE_INDEX  = 150
SCORE_THRESH  = 0.55
IMG_SIZE      = (1024, 1024) # must match training size
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # Create dataset
    cams = [Camera(name) for name in CAMERAS]
    transforms = v2.Compose([
        v2.Resize(IMG_SIZE, antialias=True)
    ])
    dataset = MultiCamDataset(
        base_dir=DATA_DIR,
        cameras=cams,
        set_type=SETTYPE,
        transforms=transforms
    )

    # Create dataloader
    dataloader = MultiCamDataloader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    # Create model and load checkpoint
    model = MultiCamModel(cameras=CAMERAS, num_classes=NUM_CLASSES, pretrained=False)
    model = model.load_checkpoint(CHECKPOINT, num_classes=NUM_CLASSES, map_location=DEVICE)
    model.to(DEVICE).eval()

    print("Starting evaluation ...")
    with torch.no_grad():
        for i, (img, target, img_path) in enumerate(tqdm(dataloader, total=len(dataloader), desc="Inference", unit="img")):


















    # Grab a sample
    sample_data = dataset[SAMPLE_INDEX]
    data = {cam: [sample] for cam, sample in sample_data.items()}
    # dataloader = MultiCamDataloader(dataset, batch_size=1, shuffle=False, num_workers=0)
    # data = next(iter(dataloader))
    inputs, targets = {}, {}
    for cam, sample in data.items():
        imgs, targs = zip(*sample)
        inputs[cam] = [img.to(DEVICE) for img in imgs]
        targets[cam] = [{k: v.to(DEVICE) for k, v in t.items()} for t in targs]

    with torch.no_grad():
        outputs = model(inputs)

    # Filter out the low scoring predictions
    preds = outputs[0]
    keep = preds["scores"].cpu() >= SCORE_THRESH
    preds["boxes"]  = preds["boxes"][keep]
    preds["labels"] = preds["labels"][keep]

    # Visualize ground‑truth vs filtered predictions
    plot(
      [
        [(inputs["cam0"][0].cpu(), targets["cam0"][0])],
        [(inputs["cam0"][0].cpu(), preds)]
      ],
      row_title=["Ground Truth", "Predictions"],
      class_names=CLASS_NAMES,
    )


if __name__ == "__main__":
    main()
