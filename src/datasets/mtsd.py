import json
import os

import torch

import tqdm

from PIL import Image

from .common import ImageFolderWithPaths, SubsetSampler

class MTSD:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=32):
        self.preprocess = preprocess
        self.location = location
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.populate_train()
        self.populate_test()

        idx_to_class = dict((v, k) for k, v in self.train_dataset.class_to_idx.items())
        self.classnames = [idx_to_class[i].replace('_', ' ').replace('-', ' ') for i in range(len(idx_to_class))]



    def populate_train(self):
        traindir = os.path.join(self.location, 'mtsd', 'train')
        self.train_dataset = ImageFolderWithPaths(
            traindir, transform=self.preprocess)
        kwargs = {'shuffle' : True} if sampler is None else {}
        print('kwargs is', kwargs)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            sampler=None,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            **kwargs,
        )

    def populate_test(self):
        self.test_dataset = self.get_test_dataset()
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=None
        )

    def get_test_path(self):
        test_path = os.path.join(self.location, 'mtsd', 'val')
        return test_path

    def get_test_dataset(self):
        return ImageFolderWithPaths(self.get_test_path(), transform=self.preprocess)



def preprocess_mtsd():
    data_dir = '~/datasets/mtsd'

    annot_dir = os.path.join(data_dir, 'annotations', 'annotations')

    classes = set()
    count_pano = 0
    count_no_pano = 0
    count = 0

    splits = ['train', 'val']

    for split in splits:
        print('='*100)
        print(split)

        with open(os.path.join(data_dir, f'annotations/splits/{split}.txt'), 'r') as f:
            info = [s.strip() for s in f.readlines()]
        
        for img_id in tqdm.tqdm(info):
            annotation_file = os.path.join(annot_dir, img_id + '.json')
            d = json.load(open(annotation_file, 'r'))
            if d['ispano']:
                continue
            
            for o in d['objects']:
                label = o['label']
                if label == 'other-sign':
                    continue
                
                if '--' in label:
                    label = label.split('--')[1]
                
                dirname = os.path.join(data_dir, split, label)
                os.makedirs(dirname, exist_ok=True)

                bb = o['bbox']
                bb = [round(bb['xmin']), round(bb['ymin']), round(bb['xmax']), round(bb['ymax'])]

                im = Image.open(os.path.join(data_dir, 'images', img_id + '.jpg'))
                im = im.crop(bb)

                im.save(os.path.join(dirname, img_id + '__' +  o['key'] + '.jpg'))