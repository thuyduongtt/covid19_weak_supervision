from PIL import Image
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn


def test_img():
    gt = Image.open('gt.bmp')
    tgt_mask = np.array(gt)

    m = tgt_mask.min()
    n = tgt_mask.max()
    mid = (m + n) / 2
    tgt_mask[tgt_mask < mid] = 0
    tgt_mask[tgt_mask >= mid] = 255

    print(tgt_mask[100:120, 100:120])

    gt_bin = Image.fromarray(tgt_mask)
    gt_bin.save('gt_bin.bmp')

    print(np.unique(tgt_mask))
    print(np.setdiff1d(np.unique(tgt_mask), [0, 127, 255]))


def testConv():
    img = torch.rand([1, 6, 256, 256])
    layer = nn.Conv2d(6, 64, kernel_size=(3, 3), stride=(1, 1), padding=(100, 100))
    out_im = layer(img)
    print(layer)
    print(out_im.size())


if __name__ == '__main__':
    testConv()
