import os
import torch
# from torchvision.datasets import Kitti as PyTorchKITTI
import numpy as np

import csv
import os
from typing import Any, Callable, List, Optional, Tuple

from PIL import Image
import zipfile

from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets.vision import VisionDataset

import collections

_VALIDATION_SPLIT_PERCENT_VIDEOS = 10
_TEST_SPLIT_PERCENT_VIDEOS = 10


def _build_splits(devkit):
    """Splits the train data into train/val/test by video.
    Ensures that images from the same video do not traverse the splits.
    Args:
        devkit: object that iterates over the devkit archive.
    Returns:
        train_images: File ids for the training set images.
        validation_images: File ids for the validation set images.
        test_images: File ids for the test set images.
    """
    mapping_line_ids = None
    mapping_lines = None
    with zipfile.ZipFile(devkit, 'r') as f:
        for fpath in f.namelist():
            if fpath == os.path.join("mapping", "train_rand.txt"):
                # Converts 1-based line index to 0-based line index.
                mapping_line_ids = [
                    int(x.strip()) - 1 for x in f.read(fpath).decode("utf-8").split(",")
                ]
            elif fpath == os.path.join("mapping", "train_mapping.txt"):
                mapping_lines = f.read(fpath).splitlines()
                mapping_lines = [x.decode("utf-8") for x in mapping_lines]

    assert mapping_line_ids
    assert mapping_lines

    video_to_image = collections.defaultdict(list)
    for image_id, mapping_lineid in enumerate(mapping_line_ids):
        line = mapping_lines[mapping_lineid]
        video_id = line.split(" ")[1]
        video_to_image[video_id].append(image_id)

    # Sets numpy random state.
    numpy_original_state = np.random.get_state()
    np.random.seed(seed=123)

    # Max 1 for testing.
    num_test_videos = max(1, _TEST_SPLIT_PERCENT_VIDEOS * len(video_to_image) // 100)
    num_validation_videos = max(
        1, _VALIDATION_SPLIT_PERCENT_VIDEOS * len(video_to_image) // 100)
    test_videos = set(
        np.random.choice(
            sorted(list(video_to_image.keys())), num_test_videos, replace=False))
    validation_videos = set(
        np.random.choice(
            sorted(list(set(video_to_image.keys()) - set(test_videos))),
            num_validation_videos,
            replace=False))
    test_images = []
    validation_images = []
    train_images = []
    for k, v in video_to_image.items():
        if k in test_videos:
            test_images.extend(v)
        elif k in validation_videos:
            validation_images.extend(v)
        else:
            train_images.extend(v)

    # Resets numpy random state.
    np.random.set_state(numpy_original_state)
    return {
        'train': train_images,
        'val': validation_images,
        'test': test_images,
    }


class KittiDistance(VisionDataset):
    """`KITTI <http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark>`_ Dataset.

    It corresponds to the "left color images of object" dataset, for object detection.

    Args:
        root (string): Root directory where images are downloaded to.
            Expects the following folder structure if download=False:

            .. code::

                <root>
                    └── Kitti
                        └─ raw
                            ├── training
                            |   ├── image_2
                            |   └── label_2
                            └── testing
                                └── image_2
        train (bool, optional): Use ``train`` split if true, else ``test`` split.
            Defaults to ``train``.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample
            and its target as entry and returns a transformed version.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    data_url = "https://s3.eu-central-1.amazonaws.com/avg-kitti/"
    resources = [
        "data_object_image_2.zip",
        "data_object_label_2.zip",
        "devkit_object.zip"
    ]
    image_dir_name = "image_2"
    labels_dir_name = "label_2"

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        download: bool = False,
        split: str = 'train',
    ):
        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        )
        self.images = []
        self.targets = []
        self.root = root
        self.train = train
        self._location = "training"
        self.split = split

        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError("Dataset not found. You may use download=True to download it.")

        self.image_ids = _build_splits(os.path.join(self._raw_folder, "devkit_object.zip"))[self.split]

        image_dir = os.path.join(self._raw_folder, self._location, self.image_dir_name)
        labels_dir = os.path.join(self._raw_folder, self._location, self.labels_dir_name)
        for img_id in self.image_ids:
            img_id = f'{img_id:06d}.png'
            self.images.append(os.path.join(image_dir, img_id))
            self.targets.append(os.path.join(labels_dir, f"{img_id.split('.')[0]}.txt"))


    # See https://github.com/openai/CLIP/issues/86
    def __getitem__(self, index):
        image = Image.open(self.images[index])
        target = self._parse_target(index)
        if self.transforms:
            image, target = self.transforms(image, target)

        vehicles = [t for t in target if t["type"] in ['Car', 'Van', 'Truck']]
        locations = [t["location"][2] for t in vehicles] + [10000.0]
        dist = min(locations)
        thresholds = np.array([-100.0, 8.0, 20.0, 999.0])
        found = np.where((thresholds - dist) < 0)
        label = np.max(found)
        
        assert 0 <= label < 4
        return image, label


    def _parse_target(self, index: int) -> List:
        target = []
        with open(self.targets[index]) as inp:
            content = csv.reader(inp, delimiter=" ")
            for line in content:
                target.append(
                    {
                        "type": line[0],
                        "truncated": float(line[1]),
                        "occluded": int(line[2]),
                        "alpha": float(line[3]),
                        "bbox": [float(x) for x in line[4:8]],
                        "dimensions": [float(x) for x in line[8:11]],
                        "location": [float(x) for x in line[11:14]],
                        "rotation_y": float(line[14]),
                    }
                )
        return target

    def __len__(self) -> int:
        return len(self.images)

    @property
    def _raw_folder(self) -> str:
        return os.path.join(self.root, "raw")

    def _check_exists(self) -> bool:
        """Check if the data directory exists."""
        folders = [self.image_dir_name]
        return all(os.path.isdir(os.path.join(self._raw_folder, self._location, fname)) for fname in folders)

    def download(self) -> None:
        """Download the KITTI data if it doesn't exist already."""

        if self._check_exists():
            return

        os.makedirs(self._raw_folder, exist_ok=True)

        # download files
        for fname in self.resources:
            download_and_extract_archive(
                url=f"{self.data_url}{fname}",
                download_root=self._raw_folder,
                filename=fname,
            )



class KITTIBase:
    def __init__(self,
                 preprocess,
                 test_split,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16):

        modified_location = os.path.join(location, 'kitti')

        self.train_dataset = KittiDistance(
            root=modified_location,
            download=True,
            split='train',
            transform=preprocess
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )

        self.test_dataset = KittiDistance(
            root=modified_location,
            download=True,
            split=test_split,
            transform=preprocess
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        self.classnames = [
            'a photo i took of a car on my left or right side.',
            'a photo i took with a car nearby.',
            'a photo i took with a car in the distance.',
            'a photo i took with no car.',
        ]
 

class KITTI(KITTIBase):
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16):
        super().__init__(preprocess, 'test', location, batch_size, num_workers)


class KITTIVal(KITTIBase):
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16):
        super().__init__(preprocess, 'val', location, batch_size, num_workers)

