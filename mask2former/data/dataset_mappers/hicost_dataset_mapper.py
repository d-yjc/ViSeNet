import copy
import numpy as np
import cv2
import torch
from detectron2.detectron2.config import configurable
from detectron2.detectron2.data import detection_utils as utils
from detectron2.detectron2.data import transforms as T
# from Mask2Former.mask2former.cat_rel_dict import cat_dict, rel_dict

# from detectron2.detectron2.structures import Boxes, Instances, BoxMode
from pycocotools import mask as coco_mask

def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks

def build_transform_gen(cfg, is_train):
    #For simplicity, we just resize the image
    image_size = cfg.INPUT.IMAGE_SIZE  # i.e., 640
    return [T.Resize((image_size, image_size))]

class HicostDatasetMapper:
    """
    A custom dataset mapper for binary saliency instance segmentation that also loads relation annotations.
    It:
      - Reads the image.
      - Applies transforms.
      - Loads COCO-format annotations (converted from OpenPSG format) that include a "relations" field -> List of [sbj_idx, obj_idx, pred].
      - Prepares the instance segmentation targets (0 = bg, 1 = salient).
      - Adds a "relations" field to the returned dict so that model can later use it.
    """
    @configurable
    def __init__(self, is_train=True, tfm_gens=None, image_format="BGR"):
        self.is_train = is_train
        self.tfm_gens = tfm_gens if tfm_gens is not None else []
        self.image_format = image_format

    @classmethod
    def from_config(cls, cfg, is_train=True):
        tfm_gens = build_transform_gen(cfg, is_train)
        return {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "image_format": cfg.INPUT.FORMAT,
        }

    def __call__(self, dataset_dict):
        # Make a deep copy so that we do not modify the original dict.
        dataset_dict = copy.deepcopy(dataset_dict)

        # Read image
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)
        # Apply transforms
        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        image_shape = image.shape[:2]  # (height, width)
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if not self.is_train:
            # For inference, we don’t need annotations.
            return dataset_dict

        # Process annotations (assume they are in COCO format with an added "relations" field)
        if "annotations" in dataset_dict:
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(annos, image_shape)
            # First, filter empty instances while masks are still in polygon format
            instances = utils.filter_empty_instances(instances)
            
            # Now convert the polygon masks to binary mask tensors
            if hasattr(instances, "gt_masks"):
                gt_masks = instances.gt_masks
                # print(f"Type of gt_masks: {type(gt_masks)}")
                # print(f"Type of gt_masks.polygons: {type(gt_masks.polygons)}")
                gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, image_shape[0], image_shape[1])
                instances.gt_masks = gt_masks
                
            dataset_dict["instances"] = instances

        # Load relationship annotations.
        dataset_dict["relations"] = dataset_dict.get("relations", [])
        if len(dataset_dict["relations"]) > 0:
            # Suppose relations are float32 (the RelationshipModule uses a linear layer expecting float)
            relations_np = np.array(dataset_dict["relations"], dtype=np.float32)
            relations_t  = torch.from_numpy(relations_np)
            dataset_dict["relations"] = relations_t
        else:
            dataset_dict["relations"] = None

        relations = dataset_dict.get("relations", [])
        if len(relations) > 0:
            new_relations = []
            sbj_ids = []
            obj_ids = []
            pred_ids = []
            for rel in relations:
                if isinstance(rel, list) and len(rel) >= 3:
                    s, o, p = rel[0], rel[1], rel[2]
                    new_relations.append([float(s), float(o), float(p)])
                    sbj_ids.append(s)
                    obj_ids.append(o)
                    pred_ids.append(p)
            dataset_dict["relations"] = torch.tensor(new_relations, dtype=torch.float32)
            dataset_dict["subject_ids"] = sbj_ids
            dataset_dict["object_ids"] = obj_ids
            dataset_dict["predicate_ids"] = pred_ids

        return dataset_dict
