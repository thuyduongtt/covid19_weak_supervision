from PIL import Image
import numpy as np
from pathlib import Path

if __name__ == '__main__':
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

