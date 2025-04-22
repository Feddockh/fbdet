import os
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm.auto import tqdm

from dataset import MultiCamDataset, SetType
from loader import MultiCamDataloader
from model import MultiCamModel
from camera import Camera
from utils.visual import plot, plot_pr_curves


# Parameters
VIS           = False
SEED          = 42
LOAD_METRICS  = False
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(BASE_DIR, "erwiam_dataset")
SETTYPE       = SetType.VAL
CAMERAS       = ['cam0']
CHECKPOINT    = os.path.join(BASE_DIR, "checkpoints", "best.pth")
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
    torch.manual_seed(SEED)
    dataloader = MultiCamDataloader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )

    # Create model and load checkpoint
    model = MultiCamModel(cameras=CAMERAS, num_classes=NUM_CLASSES, pretrained=False)
    model = model.load_checkpoint(CHECKPOINT, num_classes=NUM_CLASSES, map_location=DEVICE)
    model.to(DEVICE).eval()

    # Initialize Mean Average Precision metric
    map_metric = MeanAveragePrecision(
        box_format = "xyxy", 
        iou_type = "bbox", 
        iou_thresholds = None, # [0.50, 0.55, ..., 0.95]
        rec_thresholds = None, # [0.01, 0.02, ..., 1.0]
        max_detection_thresholds = None, # [1, 10, 100]
        class_metrics = True, # Enable per-class metrics for mAP and mAR_100
        extended_summary = True, # Enable summary with additional metrics including IOU, precision and recall
    )

    print("Starting evaluation ...")
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader, total=len(dataloader), desc="Inference", unit="img")):
            if LOAD_METRICS:
                break

            # Move the data to the device
            inputs, targets = {}, {}
            for cam, sample in data.items():
                imgs, targs = zip(*sample)
                inputs[cam] = [img.to(DEVICE) for img in imgs]
                targets[cam] = [{k: v.to(DEVICE) for k, v in t.items()} for t in targs]

            # Forward pass for inference
            outputs = model(inputs)

            # Add the predictions to the metric
            preds = outputs[0]
            map_metric.update(preds=[preds], target=targets["cam0"])

            # Filter out the low scoring predictions
            keep = preds["scores"].cpu() >= SCORE_THRESH
            preds["boxes"]  = preds["boxes"][keep]
            preds["labels"] = preds["labels"][keep]

            # Visualize ground‑truth vs filtered predictions'
            if VIS:
                plot(
                    [(inputs["cam0"][0], preds), (inputs["cam0"][0], targets["cam0"][0])],
                    col_title=["Predictions", "Ground Truth"],
                    class_names=CLASS_NAMES,
                    save_path=os.path.join(BASE_DIR, "results", f"sample_{i}.png"),
                )
                if i >= 20:
                    break

    # Compute and save the mean average precision results
    print("Computing metrics ...")
    os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True)
    if LOAD_METRICS:
        results = torch.load(os.path.join(BASE_DIR, "results", "eval_results.pth"))
    else:
        results = map_metric.compute()
    torch.save(results, os.path.join(BASE_DIR, "results", "eval_results.pth"))
    print("Evaluation Metrics:")

    print(f"mAP@[.50:.95]        = {results['map']:.4f}")
    print(f"mAP@.50              = {results['map_50']:.4f}")
    print(f"mAP@.75              = {results['map_75']:.4f}")
    print(f"mAP@.50:.95 (small)  = {results['map_small']:.4f}")
    print(f"mAP@.50:.95 (medium) = {results['map_medium']:.4f}")
    print(f"mAP@.50:.95 (large)  = {results['map_large']:.4f}")
    print(f"mAR@1                = {results['mar_1']:.4f}")
    print(f"mAR@10               = {results['mar_10']:.4f}")
    print(f"mAR@100              = {results['mar_100']:.4f}")
    print(f"mAR@100 (small)      = {results['mar_small']:.4f}")
    print(f"mAR@100 (medium)     = {results['mar_medium']:.4f}")
    print(f"mAR@100 (large)      = {results['mar_large']:.4f}")

    # Plot the precision-recall curves
    plot_pr_curves(
        results = results,
        metric = map_metric,
        class_names = CLASS_NAMES[1:],
        max_x = 1.05,
        max_y = 1.05,
        save_dir = os.path.join(BASE_DIR, "results")
    )

if __name__ == "__main__":
    main()
