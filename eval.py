import torch
import torch.nn.functional as F
from tqdm import tqdm

from iou import iou_coeff
from iou import pres_recall

from sklearn.metrics import precision_score, recall_score

def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0
    tot_rc = 0
    tot_pr = 0
    tot_f1 = 0
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)

            if net.n_classes > 1:
                tot += F.cross_entropy(mask_pred, true_masks).item()
            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                tot += iou_coeff(pred, true_masks).item()
                tot_pr += pres_recall(true_masks, pred)[0]
                tot_rc += pres_recall(true_masks, pred)[1]
                tot_f1 += pres_recall(true_masks, pred)[2]
            pbar.update()

    net.train()
    return tot / n_val, tot_pr/n_val, tot_rc/n_val, tot_f1/n_val