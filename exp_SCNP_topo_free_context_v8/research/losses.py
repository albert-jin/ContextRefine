"""Reproducible SCNP adapters. Upstream source remains unchanged."""
from pathlib import Path
import ast
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]

def one_hot(target, num_classes):
    return F.one_hot(target[:, 0].long(), num_classes).movedim(-1, 1).float()

def load_upstream_loss():
    # Execute only the reviewed class/functions, without importing MONAI/Detectron2.
    # one_hot above replaces only MONAI's label-encoding helper.
    path = ROOT / "SCNP/experiments/Detectron2/loss.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                and n.name in {"SCNPCEDiceLoss", "_dice_loss", "_crossentropy_loss"}]
    ns = {"torch": torch, "nn": nn, "one_hot": one_hot}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), ns)
    return ns["SCNPCEDiceLoss"]()

def scnp_route(logits, target, kernel=3):
    if kernel < 1 or kernel % 2 != 1:
        raise ValueError("SCNP kernel must be positive and odd")
    if logits.shape != target.shape or logits.ndim != 4:
        raise ValueError("Expected matching BCHW logits and binary/one-hot targets")
    fg, fi = F.max_pool2d(-(logits * target + 9999 * (1-target)),
                          kernel, 1, kernel//2, return_indices=True)
    bg, bi = F.max_pool2d(logits * (1-target) - 9999 * target,
                          kernel, 1, kernel//2, return_indices=True)
    routed = -fg * target + bg * (1-target)
    indices = torch.where(target.bool(), fi, bi)
    return routed, indices

def softmax_cedice(logits, labels):
    target = one_hot(labels[:, None], logits.shape[1]).to(logits)
    probs = logits.softmax(1)
    ce = -(target * (probs + 1e-15).log()).mean((0, 2, 3)).sum()
    dice = -(2*(probs*target).sum((2,3))+1e-5).div(
        (probs.sum((2,3))+target.sum((2,3))+1e-5).clamp_min(1e-8)).mean()
    return ce + dice

def binary_cedice(logits, target):
    """One-channel candidate protocol: BCE + mean FG/BG negative soft Dice."""
    p = logits.sigmoid()
    dice = []
    for q, y in ((p, target), (1-p, 1-target)):
        dice.append((2*(q*y).sum((2,3))+1e-5)/(q.sum((2,3))+y.sum((2,3))+1e-5))
    return F.binary_cross_entropy_with_logits(logits, target) - torch.stack(dice).mean()

