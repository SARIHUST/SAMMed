import numpy as np
import torch
import pdb
from torchvision.utils import save_image
from queue import Queue

def compute_iou(box1, box2, wh=False):
    """
    compute the iou of two boxes.
    args:
        box1, box2: [xmin, ymin, xmax, ymax] (wh=False) or [xcenter, ycenter, w, h] (wh=True)
        wh: the format of coordinate.
    return:
        iou: iou of box1 and box2.
    """
    if wh == False:
        xmin1, ymin1, xmax1, ymax1 = box1
        xmin2, ymin2, xmax2, ymax2 = box2
    else:
        xmin1, ymin1 = int(box1[0] - box1[2] / 2.0), int(box1[1] - box1[3] / 2.0)
        xmax1, ymax1 = int(box1[0] + box1[2] / 2.0), int(box1[1] + box1[3] / 2.0)
        xmin2, ymin2 = int(box2[0] - box2[2] / 2.0), int(box2[1] - box2[3] / 2.0)
        xmax2, ymax2 = int(box2[0] + box2[2] / 2.0), int(box2[1] + box2[3] / 2.0)

    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    inter_x1 = np.max([xmin1, xmin2])
    inter_y1 = np.max([ymin1, ymin2])
    inter_x2 = np.min([xmax1, xmax2])
    inter_y2 = np.min([ymax1, ymax2])

    inter_area = (np.max([0, inter_x2 - inter_x1])) * (np.max([0, inter_y2 - inter_y1]))
    iou = inter_area / (area1 + area2 - inter_area + 1e-6)

    return iou

def dice_score(preds, targets):
    smooth = 1.0
    assert preds.size() == targets.size()

    iflat = preds.contiguous().view(-1)
    tflat = targets.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    dice = (2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)
    return dice

def filter_mask(mask):
    from collections import deque
    mask = mask.long()
    # save_image(mask.float(), 'before.png')
    visited = torch.zeros_like(mask)
    _, r, c = mask.shape
    
    ds = [[0, 1], [0, -1], [1, 0], [-1, 0]]
            
    max_size = 0
    # q = Queue()
    q = deque()
    for i in range(r):
        for j in range(c):
            if mask[0][i][j] == 1 and visited[0][i][j] == 0:
                indices = []
                size = 0
                # q.put((i, j))
                q.append((i, j))
                visited[0][i][j] = 1
                # while not q.empty():
                while len(q) != 0:
                    # x, y = q.get()
                    x, y = q.popleft()
                    indices.append((x, y))
                    size += 1
                    for dx, dy in ds:
                        nx, ny = x + dx, y + dy
                        if nx >= 0 and nx < r and ny >= 0 and ny < c and mask[0][nx][ny] == 1 and visited[0][nx][ny] == 0:
                            # q.put((nx, ny))
                            q.append((nx, ny))
                            visited[0][nx][ny] = 1
                max_size = max(max_size, size)
                if size == max_size:
                    for x, y in indices:
                        mask[0][x][y] = size

    mask = (mask == max_size)
    # save_image(mask.float(), 'after.png')
    if torch.max(mask > 0) == 0:
        return torch.zeros(4)
    _, y_indices, x_indices = torch.where(mask > 0)
    x_min, y_min = (x_indices.min(), y_indices.min())
    x_max, y_max = (x_indices.max(), y_indices.max())
    # pdb.set_trace()
    return torch.tensor([x_min, y_min, x_max, y_max])