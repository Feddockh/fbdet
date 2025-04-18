import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.transforms.v2 import functional as F
from torchvision.tv_tensors import BoundingBoxes

def plot(
    imgs,
    row_title=None,
    class_names=None,              # list mapping class IDs → names (idx 0 = background)
    score_fmt="{:.2f}",
    **imshow_kwargs,
):
    """
    imgs        : single list or 2‑D grid of images or (img, target) tuples
    class_names : optional list of names, e.g. ["bg","apple","leaf",…]
    """
    # ───── colour palette ─────────────────────────────────────────────────────
    _palette = [
        "red","green","blue","cyan","magenta","yellow",
        "orange","purple","pink","lime","brown","gray",
    ]
    def _colour(label: int):
        return _palette[label % len(_palette)]

    def _to_uint8(img: torch.Tensor) -> torch.Tensor:
        if img.dtype.is_floating_point and img.min() < 0:
            img = (img - img.min()) / (img.max() - img.min()).clamp(min=1e-5)
        return F.to_dtype(img, torch.uint8, scale=True)

    # ───── ensure 2‑D grid ────────────────────────────────────────────────────
    if not isinstance(imgs[0], list):
        imgs = [imgs]

    n_rows, n_cols = len(imgs), len(imgs[0])
    fig, axs = plt.subplots(n_rows, n_cols, squeeze=False, figsize=(4*n_cols, 4*n_rows))

    # ───── draw each cell ─────────────────────────────────────────────────────
    for r, row in enumerate(imgs):
        for c, sample in enumerate(row):
            img = sample
            boxes = masks = labels_t = scores = None

            if isinstance(sample, tuple):
                img, tgt = sample
                if isinstance(tgt, dict):
                    boxes   = tgt.get("boxes")
                    masks   = tgt.get("masks")
                    labels_t= tgt.get("labels")
                    scores  = tgt.get("scores")
                elif isinstance(tgt, BoundingBoxes):
                    boxes = tgt
                else:
                    raise ValueError(f"Unexpected target type: {type(tgt)}")

            img = _to_uint8(F.to_image(img))

            # masks
            if masks is not None:
                img = draw_segmentation_masks(
                    img, masks.to(torch.bool),
                    colors=[_colour(i) for i in range(len(masks))],
                    alpha=0.5
                )

            # boxes + labels + scores
            if boxes is not None and len(boxes):
                if hasattr(boxes, "tensor"):
                    boxes = boxes.tensor
                if boxes.dtype.is_floating_point:
                    boxes = boxes.round().to(torch.int)
                else:
                    boxes = boxes.to(torch.int64)

                # prepare names & colours
                lbls, cols = [], []
                labels_list = labels_t.tolist() if labels_t is not None else [0]*len(boxes)
                scores_list = scores.tolist() if scores is not None else [None]*len(boxes)

                for idx, lab in enumerate(labels_list):
                    # class name
                    if class_names and lab < len(class_names):
                        nm = class_names[lab]
                    else:
                        nm = str(lab)
                    # score
                    sc = scores_list[idx]
                    if sc is not None:
                        nm += f":{score_fmt.format(sc)}"

                    lbls.append(nm)
                    cols.append(_colour(lab))

                img = draw_bounding_boxes(
                    img, boxes,
                    labels=lbls,
                    colors=cols,
                    width=3,
                    font_size=20,            # <- bigger text
                )

            ax = axs[r, c]
            ax.imshow(img.permute(1, 2, 0).cpu().numpy(), **imshow_kwargs)
            ax.set(xticks=[], yticks=[])

    # ───── row titles ─────────────────────────────────────────────────────────
    if row_title is not None:
        for r in range(n_rows):
            axs[r, 0].set_ylabel(row_title[r], rotation=0, labelpad=40, va="center")

    # ───── legend/key ─────────────────────────────────────────────────────────
    if class_names:
        handles = [
            mpatches.Patch(color=_colour(i), label=class_names[i])
            for i in range(len(class_names))
        ]
        # place legend in the figure (not per‐axis), top‐right corner
        fig.legend(
            handles=handles,
            loc="upper right",
            title="Classes",
            fontsize=12,
            title_fontsize=14,
            frameon=False,
        )
        # leave room for legend on the right
        plt.tight_layout(rect=[0, 0, 0.85, 1])
    else:
        plt.tight_layout()

    plt.show()
