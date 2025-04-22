import os
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from camera import Camera
from dataset import MultiCamDataset, SetType, DATA_DIR


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

class MultiCamDataloader(DataLoader):
    def __init__(self, dataset, batch_size=2, shuffle=False, num_workers=2):
        """
        Custom DataLoader for MultiCamDataset that uses a custom collate function to handle
        the multi-camera data structure.
        """
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, 
                         num_workers=num_workers, collate_fn=multicam_collate_fn)

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

    # Create the DataLoader using the custom collate function
    dataloader = MultiCamDataloader(dataset, batch_size=2, shuffle=True, num_workers=0)

    # Iterate through the DataLoader. In this case 'batch' is a dictionary where each key is a camera name
    # and the corresponding value is a list of (image, target) tuples for that batch.
    for batch in dataloader:
        for cam, data_list in batch.items():
            print(f"Camera: {cam}, Number of samples: {len(data_list)}")
        break

if __name__ == "__main__":
    demo()
