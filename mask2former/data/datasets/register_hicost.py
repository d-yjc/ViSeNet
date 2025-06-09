import json

from pathlib import Path
from detectron2.detectron2.structures import BoxMode

DATASET_NAME = "hicost_"
ANN_DIR = "ann" # Annotation (JSON) directory.
IMG_DIR = "img" # Image directory.

def get_dict(root, split):
    dataset_dict = []

    root = Path(root)
    json_path = root / ANN_DIR / f"{split}.json"

    with open(json_path) as file:
        annotations = json.load(file)
    
    for i, img in enumerate(annotations["data"]):
        entry = {}
        instances = []

        entry["file_name"] = str(root / IMG_DIR / split / img["file_name"][-16:]) # Omit COCO split prefix (i.e. train2017, val2017)
        entry["image_id"] = img["image_id"]
        entry["width"] = img["width"]
        entry["height"] = img["height"]

        for anno, segment in zip(img["annotations"], img["segments_info"]):
            if (segment["isthing"] == 1 and segment["iscrowd"] == 0):
                instance = {
                "bbox": anno["bbox"],
                "bbox_mode": BoxMode.XYXY_ABS, # all OpenPSG images have bbox_mode = 0, may need to convert to xywh ...
                "category_id": 0,
                "segmentation": segment["segmentation"]
                } 
                instances.append(instance)
        entry["annotations"] = instances
        entry["relations"] = img["relations"]

        dataset_dict.append(entry)
    return dataset_dict[:]
