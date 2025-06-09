import sys
import os
import logging
from detectron2.detectron2.config import get_cfg
from detectron2.detectron2.engine import default_argument_parser, launch
from detectron2.detectron2.modeling import build_model
from detectron2.detectron2.checkpoint import DetectionCheckpointer
import detectron2.detectron2.utils.comm as comm
from inference import inference  # import the inference function from inference.py

from mask2former import add_maskformer2_config
from detectron2.projects.DeepLab.deeplab import add_deeplab_config

logger = logging.getLogger("detectron2")

def setup(args):
    cfg = get_cfg()
    # Allow new keys to be added.
    cfg.set_new_allowed(True)
    # Register custom keys from DeepLab and Mask2Former.
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    
    cfg.freeze()
    return cfg

def main(args):
    cfg = setup(args)
    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    model_dir = cfg.EVALUATION.MODEL_DIR
    split = "test"

    if comm.is_main_process():
        # Load model weights
        DetectionCheckpointer(model, save_dir=model_dir).resume_or_load(model_dir, resume=args.resume)
    
    # Run inference. This returns a tuple of metrics and the results list.
    results = inference(cfg, model, split)
    

        

if __name__ == "__main__":
    parser = default_argument_parser()
    args = parser.parse_args()
    args.config_file = './configs/coco/instance-segmentation/swin/maskformer2_swin_tiny_bs16_50ep.yaml'
    args.resume = False
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=1,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
