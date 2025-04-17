import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from pycocotools.coco import COCO
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import decode_image
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F
from torchvision.tv_tensors import Image, BoundingBoxes, BoundingBoxFormat, Mask
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from utils import Camera


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DATA_DIR = os.path.join(BASE_DIR, "rivendale_dataset")
DATA_DIR = os.path.join(BASE_DIR, "erwiam_dataset")

class SetType:
    ALL = "annotations"
    TRAIN = "train"
    VAL = "val"

class MultiCamDataset(Dataset):
    def __init__(self, base_dir: str, cameras: List[Camera], 
                 set_type: SetType = SetType.ALL, transforms = None):
        """
        Initialize the multi-cam dataset with the base directory and camera names.
        All cameras must have the same number of images.
        """
        self.base_dir = base_dir
        self.cameras = cameras
        self.transforms = transforms
        self.annotations: Dict[str, COCO] = {}

        # Load annotations from COCO format for each camera
        for cam in cameras:
            cam_dir = os.path.join(base_dir, cam.name)
            if not os.path.exists(cam_dir):
                raise ValueError(f"Camera directory {cam_dir} does not exist.")
            cam_annotations_path = os.path.join(cam_dir, set_type + '.json')
            self.annotations[cam.name] = COCO(cam_annotations_path)

        # Load the image ids from the first camera (should be the same for all cameras)
        self.ids = self.annotations[cameras[0].name].getImgIds()

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset. 
        
        This returns a dictionary with the camera name as the key corresponding to 
        each cameras matching (image, annotations) tuple.

        Each camera has its own image and annotations that correspond to the same 
        capture instance. The images are loaded in RGB format and the annotations 
        are loaded in COCO format. The annotations are converted to tv tensors
        so that they can be transformed using the same transforms as the images.

        The annotations are a dictionary with keys 'boxes', 'masks', and 'labels'.
        The 'boxes' are in XYXY format, the 'masks' are in binary format, and the
        'labels' are in integer format.
        """
        sample: Dict[str, (Image, Dict)] = {}

        # Get the image id and filename (same for all cameras)
        img_id = self.ids[idx]
        img_filename = self.annotations[self.cameras[0].name].imgs[img_id]['file_name']

        # Iterate through each camera
        for cam in self.cameras:
            
            ## Load the image ##
            # Formulate the full path for the image (file paths should be the same except for the camera name)
            img_path = os.path.join(self.base_dir, cam.name, "images", img_filename)
            # Check if the image file exists
            if not os.path.exists(img_path):
                raise ValueError(f"Image file {img_path} does not exist.")
            # Load in the image as a tensor with dimensions [C, H, W]
            img = decode_image(img_path)

            ## Load the annotations ##
            # Get the annotations for this image
            ann_ids = self.annotations[cam.name].getAnnIds(imgIds=img_id)
            anns = self.annotations[cam.name].loadAnns(ann_ids)

            # Define the canvas size as (height, width)
            canvas_size = (img.shape[1], img.shape[2])

            # If no annotations exist for this image, create empty targets.
            if len(anns) == 0:
                boxes_tv = BoundingBoxes(torch.empty((0, 4), dtype=torch.float32),
                                        format=BoundingBoxFormat.XYWH,
                                        canvas_size=canvas_size)
                masks_tv = Mask(torch.empty((0, canvas_size[0], canvas_size[1]), dtype=torch.uint8))
                labels_tensor = torch.empty((0,), dtype=torch.uint8)
            else:
                ## Format the annotations as tv tensor ##
                # Each annotation dict contains a key "bbox" with [x, y, width, height]
                boxes_list = [ann['bbox'] for ann in anns]
                # Convert the list of boxes to a tensor of shape [num_boxes, 4]
                boxes_tensor = torch.tensor(boxes_list, dtype=torch.float32)
                # Create the BoundingBoxes TVTensor with the boxes in XYWH format
                boxes_tv = BoundingBoxes(boxes_tensor, format=BoundingBoxFormat.XYWH, canvas_size=canvas_size)
                # Convert the boxes to XYXY format using transforms v2 for tv tensors
                boxes_tv = F.convert_bounding_box_format(boxes_tv, new_format=BoundingBoxFormat.XYXY)

                # Each annotation dict contains a key "segmentation" with lists of polygons
                coco = self.annotations[cam.name]
                # Iterate through each annotation and convert the segmentation to a binary mask
                masks_list = []
                for ann in anns:
                    if ann['segmentation'] != []:
                        masks_list.append(coco.annToMask(ann))
                    else:
                        masks_list.append(np.zeros((img.shape[1], img.shape[2]), dtype=np.uint8))
                # Convert the list to a single numpy array of shape [num_masks, height, width]
                masks_np = np.array(masks_list, dtype=np.uint8)
                # Convert the list of masks to a tensor of shape [num_masks, height, width]
                masks_tensor = torch.tensor(masks_np, dtype=torch.uint8)
                # Create the Mask TVTensor with the masks
                masks_tv = Mask(masks_tensor)

                # Each annotation dict contains a key "category_id" with the class label
                labels_list = [ann['category_id'] for ann in anns]
                # Convert the list of labels to a tensor of shape [num_labels]
                labels_tensor = torch.tensor(labels_list, dtype=torch.uint8)

            # Create a dictionary of annotations
            target = {
                'boxes': boxes_tv,
                'masks': masks_tv,
                'labels': labels_tensor
            }

            # Apply transform if provided (can apply to both image and annotations)
            # due to the use of torchvision.transforms.v2 and torchvision.tv_tensors
            if self.transforms:
                img, target = self.transforms(img, target)

            # Add the image and annotations to the sample for this camera
            sample[cam.name] = (img, target)
        return sample
    
# Source: https://github.com/pytorch/vision/blob/main/gallery/transforms/helpers.py
def plot(imgs, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            boxes = None
            masks = None
            if isinstance(img, tuple):
                img, target = img
                if isinstance(target, dict):
                    boxes = target.get("boxes")
                    masks = target.get("masks")
                elif isinstance(target, BoundingBoxes):
                    boxes = target
                else:
                    raise ValueError(f"Unexpected target type: {type(target)}")
            img = F.to_image(img)
            if img.dtype.is_floating_point and img.min() < 0:
                # Poor man's re-normalization for the colors to be OK-ish. This
                # is useful for images coming out of Normalize()
                img -= img.min()
                img /= img.max()

            img = F.to_dtype(img, torch.uint8, scale=True)
            if boxes is not None:
                img = draw_bounding_boxes(img, boxes, colors="yellow", width=3)
            if masks is not None:
                img = draw_segmentation_masks(img, masks.to(torch.bool), colors=["green"] * masks.shape[0], alpha=.65)

            ax = axs[row_idx, col_idx]
            ax.imshow(img.permute(1, 2, 0).numpy(), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
    plt.show()
    
def multicam_collate_fn(samples):
    """
    Collate function that takes a list of samples (each a dict mapping camera names to (image, target) tuples)
    and returns a single dict where each camera name maps to a list of (image, target) tuples.
    """
    batch = {}
    for sample in samples:
        for cam_name, data in sample.items():
            batch.setdefault(cam_name, []).append(data)
    return batch

def demo():
    # Create the cameras
    # cam0 = Camera("firefly_left") # Use this for the rivendale dataset
    cam0 = Camera("cam0") # Use this for the erwiam dataset
    cameras = [cam0]

    # Define the transforms
    transforms = v2.Compose([
        v2.Resize((1024, 1024), antialias=True), # Higher for finer details
        v2.RandomHorizontalFlip(p=0.5),
    ])

    # Create the dataset
    dataset = MultiCamDataset(DATA_DIR, cameras, set_type=SetType.TRAIN, transforms=transforms)
    view_idx = 100 # Make sure this index is valid for your dataset
    img, target = dataset[view_idx][cam0.name] 

    # Print the shape of the image and annotations and plot the image with annotations
    print(f"Image shape: {img.shape}")
    print(f"Annotation boxes shape: {target['boxes'].shape}")
    print(f"Annotation masks shape: {target['masks'].shape}")
    print(f"Annotation labels shape: {target['labels'].shape}")
    print(f"Annotation labels: {target['labels']}")
    plot([(img, target)])

    # Create the DataLoader using the custom collate function
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=multicam_collate_fn)

    # Iterate through the DataLoader. In this case 'batch' is a dictionary where each key is a camera name
    # and the corresponding value is a list of (image, target) tuples for that batch.
    for batch in dataloader:
        for cam, data_list in batch.items():
            print(f"Camera: {cam}, Number of samples: {len(data_list)}")
        break

if __name__ == "__main__":
    demo()
