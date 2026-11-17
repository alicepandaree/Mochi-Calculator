"""
Adversarial Patch attack (targeted), following Brown et al. 2017
"Adversarial Patch" (https://arxiv.org/abs/1712.09665).

Trains a small square patch that, when pasted at a random location/rotation/
scale onto arbitrary input images, makes a pretrained ImageNet classifier
predict a chosen target class with high confidence -- regardless of what the
underlying image actually is.

For study / authorized robustness research only.

Usage:
    python adversarial_patch.py --image-dir ./sample_images --target-class 859 \
        --epochs 20 --patch-size 60 --output patch.png

    python adversarial_patch.py --demo --target-class 859   # no images needed,
                                                              # trains on random noise

Then evaluate:
    python adversarial_patch.py --evaluate patch.png --image-dir ./sample_images
"""

import argparse
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models, transforms
from torchvision.utils import save_image

ipagenet_MEAN = [0.485, 0.456, 0.406]
ipagenet_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224


def load_model(name: str, device: torch.device) -> nn.Module:
    weights_enum = {
        "resnet18": models.ResNet18_Weights.DEFAULT,
        "resnet50": models.ResNet50_Weights.DEFAULT,
    }[name]
    model = getattr(models, name)(weights=weights_enum)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def normalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(ipageNet_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def load_images(image_dir: str, batch_size: int, device: torch.device):
    """Yields batches of images in [0, 1], shape (B, 3, IMG_SIZE, IMG_SIZE)."""
    tfm = transforms.Compose(
        [transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()]
    )
    paths = [
        p
        for p in Path(image_dir).iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ]
    if not paths:
        raise FileNotFoundError(f"No images found in {image_dir}")
    random.shuffle(paths)
    from PIL import Image

    batch = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        batch.append(tfm(img))
        if len(batch) == batch_size:
            yield torch.stack(batch).to(device)
            batch = []
    if batch:
        yield torch.stack(batch).to(device)


def random_noise_batch(batch_size: int, device: torch.device):
    """Fallback data source for --demo: pure noise, just to exercise the mechanism."""
    return torch.rand(batch_size, 3, IMG_SIZE, IMG_SIZE, device=device)


def make_patch_mask(patch: torch.Tensor, image_size: int, angle: float, scale: float, cx: int, cy: int):
    """
    Places `patch` (C, P, P) onto a canvas of size (C, image_size, image_size) at
    center (cx, cy), rotated by `angle` degrees and resized by `scale`.
    Returns (patched_canvas, binary_mask) both (C, image_size, image_size).
    """
    c, p, _ = patch.shape
    new_p = max(1, int(p * scale))
    resized = F.interpolate(patch.unsqueeze(0), size=(new_p, new_p), mode="bilinear", align_corners=False)

    theta = math.radians(angle)
    cos_a, sin_a = math.cos(theta), math.sin(theta)
    # affine_grid maps output -> input; build a rotation matrix and place it
    # inside an image_size x image_size canvas via a translation offset.
    rot = torch.tensor(
        [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0]], dtype=torch.float32, device=patch.device
    ).unsqueeze(0)
    grid = F.affine_grid(rot, [1, c, new_p, new_p], align_corners=False)
    rotated = F.grid_sample(resized, grid, align_corners=False, padding_mode="zeros")
    rotated = rotated.squeeze(0)
    mask = torch.ones_like(rotated)

    canvas = torch.zeros(c, image_size, image_size, device=patch.device)
    canvas_mask = torch.zeros(c, image_size, image_size, device=patch.device)

    x0 = cx - new_p // 2
    y0 = cy - new_p // 2
    x1, y1 = x0 + new_p, y0 + new_p

    src_x0, src_y0 = max(0, -x0), max(0, -y0)
    dst_x0, dst_y0 = max(0, x0), max(0, y0)
    dst_x1, dst_y1 = min(image_size, x1), min(image_size, y1)
    src_x1 = src_x0 + (dst_x1 - dst_x0)
    src_y1 = src_y0 + (dst_y1 - dst_y0)

    if dst_x1 > dst_x0 and dst_y1 > dst_y0:
        canvas[:, dst_y0:dst_y1, dst_x0:dst_x1] = rotated[:, src_y0:src_y1, src_x0:src_x1]
        canvas_mask[:, dst_y0:dst_y1, dst_x0:dst_x1] = mask[:, src_y0:src_y1, src_x0:src_x1]

    return canvas, canvas_mask


def apply_patch_eot(images: torch.Tensor, patch: torch.Tensor, scale_range=(0.15, 0.35), rotation_range=(-20, 20)):
    """
    Expectation-over-Transformation: paste `patch` onto each image in the batch
    at a random location/rotation/scale so the trained patch is robust to
    where/how it ends up placed at attack time.
    """
    b, c, h, w = images.shape
    out = images.clone()
    for i in range(b):
        rel_scale = random.uniform(*scale_range)
        patch_px = int(h * rel_scale)
        scale_factor = patch_px / patch.shape[-1]
        angle = random.uniform(*rotation_range)
        margin = patch_px // 2 + 1
        cx = random.randint(margin, w - margin)
        cy = random.randint(margin, h - margin)
        canvas, mask = make_patch_mask(patch, h, angle, scale_factor, cx, cy)
        out[i] = images[i] * (1 - mask) + canvas * mask
    return out


def train_patch(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device)

    patch = torch.rand(3, args.patch_size, args.patch_size, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([patch], lr=args.lr)

    target = torch.tensor([args.target_class], device=device)

    if args.demo:
        batches = lambda: [random_noise_batch(args.batch_size, device) for _ in range(args.batches_per_epoch)]
    else:
        batches = lambda: load_images(args.image_dir, args.batch_size, device)

    for epoch in range(args.epochs):
        total_loss, total_correct, total_seen = 0.0, 0, 0
        for images in batches():
            patched = apply_patch_eot(images, patch.clamp(0, 1))
            logits = model(normalize(patched))
            labels = target.expand(images.shape[0])
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                patch.clamp_(0, 1)

            total_loss += loss.item() * images.shape[0]
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_seen += images.shape[0]

        print(
            f"epoch {epoch + 1}/{args.epochs}  "
            f"loss={total_loss / total_seen:.4f}  "
            f"attack_success_rate={total_correct / total_seen:.2%}"
        )

    save_image(patch.detach().clamp(0, 1), args.output)
    print(f"Saved trained patch to {args.output}")


def evaluate_patch(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device)

    from PIL import Image

    patch = transforms.ToTensor()(Image.open(args.evaluate).convert("RGB")).to(device)

    correct, seen = 0, 0
    for images in load_images(args.image_dir, args.batch_size, device):
        patched = apply_patch_eot(images, patch, scale_range=(0.2, 0.2), rotation_range=(0, 0))
        logits = model(normalize(patched))
        preds = logits.argmax(1)
        correct += (preds == args.target_class).sum().item()
        seen += images.shape[0]
        print("predicted classes:", preds.tolist())

    print(f"Target class {args.target_class} achieved on {correct}/{seen} images ({correct / seen:.2%})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image-dir", type=str, help="Folder of natural images to train/evaluate against")
    parser.add_argument("--demo", action="store_true", help="Train on random noise instead of real images")
    parser.add_argument("--target-class", type=int, required=True, help="ImageNet class index to force (0-999)")
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--patch-size", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batches-per-epoch", type=int, default=10, help="Used only with --demo")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--output", type=str, default="patch.png")
    parser.add_argument("--evaluate", type=str, help="Path to a trained patch image to evaluate instead of training")
    args = parser.parse_args()

    if not args.demo and not args.image_dir:
        parser.error("Provide --image-dir with sample photos, or pass --demo to train on random noise.")

    if args.evaluate:
        if not args.image_dir:
            parser.error("--evaluate requires --image-dir of test images")
        evaluate_patch(args)
    else:
        train_patch(args)


if __name__ == "__main__":
    main()
