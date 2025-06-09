#!/usr/bin/env python
import sys
import argparse
from pathlib import Path
import numpy as np
import cv2
import torch

# If needed, insert your Mask2Former project path
sys.path.insert(0, "Mask2Former")

# Detectron2 / Mask2Former imports
from detectron2.detectron2.config import get_cfg
from detectron2.detectron2.engine import DefaultPredictor
from detectron2.detectron2.data import MetadataCatalog
from detectron2.projects.DeepLab.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config

CONFIDENCE = 0.91

def setup_cfg():
    """
    Load config for Mask2Former instance segmentation (saliency).
    Update paths to your config and weights as needed.
    """
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)

    # Example: a typical instance-seg config for Mask2Former
    cfg.merge_from_file("configs/coco/instance-segmentation/swin/maskformer2_swin_tiny_bs16_50ep.yaml")
    cfg.MODEL.WEIGHTS = "./output/model_final_v2.pth"  # path to your trained weights

    # We want instance segmentation
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False

    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg

def main(image_path: str, outdir_path: str):
    # Set up configuration
    cfg = setup_cfg()
    predictor = DefaultPredictor(cfg)

    # metadata for the test dataset
    # e.g. "hicost_test" with 1 class
    MetadataCatalog.get("hicost_test").set(thing_classes=["salient_obj"])

    # Load image
    im = cv2.imread(image_path)
    if im is None:
        raise ValueError(f"Could not load image from {image_path}")
    h, w = im.shape[:2]
    img_area = h * w
    # Define the maximum allowed area as a fraction of the image area.
    # For example, max_area_ratio=0.3 means we filter out masks that cover more than 30% of the image.
    max_area_ratio = 0.3
    max_area_threshold = img_area * max_area_ratio

    # Get 'Instances' from 'outputs' and Run inference
    outputs = predictor(im)

    instances = outputs["instances"].to("cpu")
    # Filter out non-salient and low confidence instances.
    class_keep = (instances.pred_classes == 0)
    confidence_keep = (instances.scores >= CONFIDENCE)
        
    keep = class_keep & confidence_keep

    filtered_instances = instances[keep]

    # Filter out large 'stuff' objects.
    areas = filtered_instances.pred_masks.sum(dim=(1, 2))
    area_keep = (areas < max_area_threshold)
    filtered_instances = filtered_instances[area_keep]

    # Print the probability/confidence of each kept instance.
    for i, score in enumerate(filtered_instances.scores):
        print(f"Instance {i} kept with confidence score: {score.item():.4f}")

    # Make sure output directory exists
    outdir = Path(outdir_path)
    outdir.mkdir(parents=True, exist_ok=True)
    image_stem = Path(image_path).stem

    # pred_masks: [N, H, W], float
    instance_masks = filtered_instances.pred_masks

    # Save each instance mask separately
    for i, mask in enumerate(instance_masks):
        # Convert to uint8 [0 or 255]
        single_mask = mask.numpy().astype(np.uint8) * 255
        single_mask_name = f"{image_stem}_instance_{i}.png"
        single_mask_path = outdir / single_mask_name
        cv2.imwrite(str(single_mask_path), single_mask)
        # print(f"Saved instance mask {i} to {single_mask_path}")

    # Combine (union) all instance masks → single saliency mask
    combined_mask = torch.any(instance_masks, dim=0).numpy().astype(np.uint8) * 255
    combined_name = f"{image_stem}_combined.png"
    combined_path = outdir / combined_name
    cv2.imwrite(str(combined_path), combined_mask)
    # print(f"Saved combined saliency mask to {combined_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--outdir", required=True, help="Directory to save all masks")
    args = parser.parse_args()

    main(args.image, args.outdir)
