from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from haven import haven_utils as hu


class QBDataset(data.Dataset):
    def __init__(self, split, datadir, exp_dict):
        path = Path(datadir, 'QB', split)
        input_dir_1 = 'input1'
        input_dir_2 = 'input2'
        label_dir = 'label'

        items = []

        for regionOrPatch in Path(path, input_dir_1).iterdir():
            if regionOrPatch.is_dir():
                for patch in regionOrPatch.iterdir():
                    items.append((
                        patch,
                        Path(path, input_dir_2, regionOrPatch.stem, patch.name),
                        Path(path, label_dir, regionOrPatch.stem, patch.name),
                        patch.name
                    ))
            else:
                items.append((
                    regionOrPatch,
                    Path(path, input_dir_2, regionOrPatch.name),
                    Path(path, label_dir, regionOrPatch.name),
                    regionOrPatch.name
                ))

        self.n_classes = exp_dict['dataset']['n_classes']
        self.images = items
        self.size = 256
        self.split = split

        self.images = sorted(self.images, key=lambda item: item[0])

        if split == 'train':
            s = 0
            e = int(0.9 * len(self.images))

        elif split == 'val':
            s = int(0.9 * len(self.images))
            e = len(self.images)

        elif split == 'test':
            s = 0
            e = len(self.images)

        self.images = self.images[s:e]
        self.dataset_size = len(self.images)

        self.img_transform = transforms.Compose([
            # transforms.Resize((self.size, self.size)),
            # transforms.rotate(-90),
            # transforms.CenterCrop((384, 385)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406, 0.485, 0.456, 0.406], [0.229, 0.224, 0.225, 0.229, 0.224, 0.225])
        ])

        if split == 'train':
            self.gt_transform = transforms.Compose([
                # transforms.Resize((self.size, self.size), interpolation=PIL.Image.NEAREST),
                # transforms.rotate(-90),
                # transforms.CenterCrop((384, 385)),
                # transforms.ToTensor()
            ])
        else:
            self.gt_transform = transforms.Compose([
                # transforms.Resize((self.size, self.size), interpolation=PIL.Image.NEAREST),
                # transforms.rotate(-90),
                # transforms.CenterCrop((384, 385)),
                # transforms.ToTensor()
            ])

    def __getitem__(self, index):
        img_path_1, img_path_2, label_path, img_name = self.images[index]

        img1 = Image.open(img_path_1).convert('RGB')
        img2 = Image.open(img_path_2).convert('RGB')
        image = np.concatenate((img1, img2), axis=-1)
        gt = Image.open(label_path)

        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        tgt_mask = np.array(gt)

        # binarize the ground truth
        m = tgt_mask.min()
        n = tgt_mask.max()
        mid = (m + n) / 2
        tgt_mask[tgt_mask < mid] = 0
        tgt_mask[tgt_mask >= mid] = 255

        assert (len(np.setdiff1d(np.unique(tgt_mask), [0, 127, 255])) == 0)

        mask = np.zeros(tgt_mask.shape)
        if self.n_classes == 2:
            mask[tgt_mask != 0] = 1

        elif self.n_classes == 3:
            mask[tgt_mask == 127] = 1
            mask[tgt_mask == 255] = 2
        mask = torch.LongTensor(mask)
        # gt = self.gt_transform(gt)

        # cc = K.CenterCrop((384, 385))
        # image = cc(image)
        # mask = cc(mask[None].float()).long()
        from src.modules.lcfcn import lcfcn_loss
        points = lcfcn_loss.get_points_from_mask(mask.numpy().squeeze(), bg_points=-1)
        points = torch.LongTensor(points)
        # hu.save_image('tmp.png', hu.denormalize(image, 'rgb'), points=points, radius=2)
        # hu.save_image('tmp.png', hu.denormalize(image, 'rgb'), mask=gt.numpy(), radius=2)
        if self.n_classes == 2:
            assert (len(np.setdiff1d(np.unique(mask), [0, 1])) == 0)
        if self.n_classes == 3:
            assert (len(np.setdiff1d(np.unique(mask), [0, 1, 2])) == 0)

        # points = cc(torch.LongTensor(points)[None].float()).long()[0]

        batch = {'images': image,
                 'masks': mask[None],
                 'points': points,
                 'meta': {'name': index,
                          'hash': hu.hash_dict({'id': self.images[index]}),
                          # 'hash':self.images[index],
                          'shape': mask.squeeze().shape,
                          'index': index,
                          'split': self.split,
                          # 'size':self.size
                          }}

        # return image, gt, name, np.array(F.interpolate(image, gt.size, mode='bilinear'))
        return batch

    def __len__(self):
        return self.dataset_size
