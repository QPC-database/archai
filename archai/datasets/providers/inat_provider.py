# Code adapted from https://github.com/macaodha/inat_comp_2018/blob/master/inat2018_loader.py

from typing import List, Tuple, Union, Optional
import os
import json

from overrides import overrides, EnforceOverrides
from torch.utils.data.dataset import Dataset

import torchvision
from torchvision.transforms import transforms

from archai.datasets.dataset_provider import DatasetProvider, register_dataset_provider, TrainTestDatasets
from archai.common.config import Config


# copied from https://github.com/macaodha/inat_comp_2018/blob/master/inat2018_loader.py
def default_loader(path):
    return Image.open(path).convert('RGB')

# copied from https://github.com/macaodha/inat_comp_2018/blob/master/inat2018_loader.py
def load_taxonomy(ann_data, tax_levels, classes):
    # loads the taxonomy data and converts to ints
    taxonomy = {}

    if 'categories' in ann_data.keys():
        num_classes = len(ann_data['categories'])
        for tt in tax_levels:
            tax_data = [aa[tt] for aa in ann_data['categories']]
            _, tax_id = np.unique(tax_data, return_inverse=True)
            taxonomy[tt] = dict(zip(range(num_classes), list(tax_id)))
    else:
        # set up dummy data
        for tt in tax_levels:
            taxonomy[tt] = dict(zip([0], [0]))

    # create a dictionary of lists containing taxonomic labels
    classes_taxonomic = {}
    for cc in np.unique(classes):
        tax_ids = [0]*len(tax_levels)
        for ii, tt in enumerate(tax_levels):
            tax_ids[ii] = taxonomy[tt][cc]
        classes_taxonomic[cc] = tax_ids

    return taxonomy, classes_taxonomic

# copied from https://github.com/macaodha/inat_comp_2018/blob/master/inat2018_loader.py
class Inat(Dataset):
    def __init__(self, root, ann_file, is_train=True):

        # load annotations
        print('Loading annotations from: ' + os.path.basename(ann_file))
        with open(ann_file) as data_file:
            ann_data = json.load(data_file)

        # set up the filenames and annotations
        self.imgs = [aa['file_name'] for aa in ann_data['images']]
        self.ids = [aa['id'] for aa in ann_data['images']]

        # if we dont have class labels set them to '0'
        if 'annotations' in ann_data.keys():
            self.classes = [aa['category_id'] for aa in ann_data['annotations']]
        else:
            self.classes = [0]*len(self.imgs)

        # load taxonomy
        self.tax_levels = ['id', 'genus', 'family', 'order', 'class', 'phylum', 'kingdom']
                           #8142, 4412,    1120,     273,     57,      25,       6
        self.taxonomy, self.classes_taxonomic = load_taxonomy(ann_data, self.tax_levels, self.classes)

        # print out some stats
        print('\t' + str(len(self.imgs)) + ' images')
        print('\t' + str(len(set(self.classes))) + ' classes')

        self.root = root
        self.is_train = is_train
        self.loader = default_loader

        # augmentation params
        self.im_size = [299, 299]  # can change this to train on higher res
        self.mu_data = [0.485, 0.456, 0.406]
        self.std_data = [0.229, 0.224, 0.225]
        self.brightness = 0.4
        self.contrast = 0.4
        self.saturation = 0.4
        self.hue = 0.25

        # augmentations
        self.center_crop = transforms.CenterCrop((self.im_size[0], self.im_size[1]))
        self.scale_aug = transforms.RandomResizedCrop(size=self.im_size[0])
        self.flip_aug = transforms.RandomHorizontalFlip()
        self.color_aug = transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        self.tensor_aug = transforms.ToTensor()
        self.norm_aug = transforms.Normalize(mean=self.mu_data, std=self.std_data)

    def __getitem__(self, index):
        path = self.root + self.imgs[index]
        im_id = self.ids[index]
        img = self.loader(path)
        species_id = self.classes[index]
        tax_ids = self.classes_taxonomic[species_id]

        if self.is_train:
            img = self.scale_aug(img)
            img = self.flip_aug(img)
            img = self.color_aug(img)
        else:
            img = self.center_crop(img)

        img = self.tensor_aug(img)
        img = self.norm_aug(img)

        return img, im_id, species_id, tax_ids

    def __len__(self):
        return len(self.imgs)




class InatProvider(DatasetProvider):
    def __init__(self, conf_data:Config):
        super().__init__(conf_data)
        self._dataroot = conf_data['dataroot']

    @overrides
    def get_datasets(self, load_train:bool, load_test:bool,
                     transform_train, transform_test)->TrainTestDatasets:
        trainset, testset = None, None

        if load_train:
            traindataroot = os.path.join(self._dataroot, 'inaturalist-2019-fgv6', 'train_val2019')
            trainfile = os.path.join(self._dataroot, 'inaturalist-2019-fgv6', 'train2019.json')
            trainset = Inat(traindataroot, trainfile, is_train=True)            
        if load_test:
            # NOTE: we are using the official val split in val2019.json as the test set. 
            # the actual test set is hidden from the public: https://www.kaggle.com/c/inaturalist-2019-fgvc6
            valdataroot = os.path.join(self._dataroot, 'inaturalist-2019-fgv6', 'train_val2019')
            valfile = os.path.join(self._dataroot, 'inaturalist-2019-fgv6', 'val2019.json')
            testset = Inat(valdataroot, valfile, is_train=False)            

        return trainset, testset

    @overrides
    def get_transforms(self)->tuple:
        # using directly from 
        # https://github.com/macaodha/inat_comp_2018/blob/master/inat2018_loader.py
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]

        # transformations match that in 
        # https://github.com/macaodha/inat_comp_2018/blob/master/inat2018_loader.py
        train_transf = [
            transforms.RandomResizedCrop(299),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.25)
        ]

        # in https://github.com/macaodha/inat_comp_2018/blob/master/inat2018_loader.py 
        # they only do a center crop at 299 for test transform
        test_transf = [transforms.CenterCrop(299)]

        normalize = [
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ]

        train_transform = transforms.Compose(train_transf + normalize)
        test_transform = transforms.Compose(test_transf + normalize)

        return train_transform, test_transform

register_dataset_provider('inaturalist', InatProvider)