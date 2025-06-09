import os
import cv2
import torch
from torch.utils.data import DataLoader
import numpy as np
from contextlib import contextmanager
import logging
import sys
from tqdm import tqdm

# Ensure detectron2 imports work
try:
    import detectron2.detectron2.utils.comm as comm
    from detectron2.detectron2.data import DatasetFromList, MapDataset
    from detectron2.detectron2.structures import Instances
except ImportError:
    print("Detectron2 imports failed. Ensure detectron2 is installed and accessible.")
    sys.exit(1)

# Ensure local imports work
try:
    from mask2former.data.datasets.register_hicost import get_dict
    from mask2former.data.dataset_mappers.hicost_dataset_mapper import HicostDatasetMapper, convert_coco_poly_to_mask
    # --- Removed metrics calculation import ---
    # from evals.cal_measures import cal_mae, cal_emeasure, cal_fmeasure, cal_smeasure
except ImportError as e:
    print(f"Import Error: {e}. Check paths.")
    sys.exit(1)

# Define dataset directory
DATASET_DIR = './mask2former/data/datasets/HiCoST/' # Adjust if your dataset path differs
# Epsilon for numerical stability (kept in case needed for future processing)
EPS = np.finfo(np.float64).eps
logger = logging.getLogger("detectron2")

def trivial_batch_collator(batch):
    """ Batch collator """
    return batch

@contextmanager
def inference_context(model):
    """ Sets model to eval mode """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)

# Modified: Removed cfg pass initially, now added back for threshold and weights info
# Modified: Returns only results_log
def run_inference_loop(dataloader, model, cfg, sal_map_out):
    """
    Runs inference loop, processes standard Detectron2 'Instances' output, saves maps.
    Does NOT calculate evaluation metrics.
    """
    results_log = []
    # --- Removed metric lists ---
    # mae_list, f_list, e_list, s_list = [], [], [], []

    # Get threshold and weights info from cfg
    confidence_threshold = cfg.EVALUATION.get("RESULT_THRESHOLD", 0.1) # Use .get for safety, provide default
    print(f"Using confidence threshold of {confidence_threshold}")
    logger.info(f"(Using Confidence Threshold of: {confidence_threshold})")
    if cfg.MODEL.WEIGHTS:
        logger.info(f"Model weights used: {cfg.MODEL.WEIGHTS}")
    else:
        logger.warning("Model weights path (cfg.MODEL.WEIGHTS) not specified in config.")


    with inference_context(model), torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Saving Saliency Maps")): # Updated description
            if not batch:
                logger.warning(f"Received empty batch at index {i}. Skipping.")
                continue

            # Assuming batch contains a single dictionary as per trivial_batch_collator
            if not isinstance(batch, list) or len(batch) != 1 or not isinstance(batch[0], dict):
                 logger.error(f"Unexpected batch format at index {i}. Expected list with one dict. Got: {type(batch)}. Skipping.")
                 results_log.append({'img_name': f'error_bad_batch_format_{i}', 'pred_masks': []})
                 continue

            input_dict = batch[0]
            model_input = [input_dict] # Model expects a list of dicts

            # --- Extract Image Info ---
            # No need for GT extraction if not calculating metrics
            try:
                img_info = input_dict
                h, w = img_info["height"], img_info["width"]
                # Use original file name for saving
                name = os.path.basename(img_info["file_name"])
                # Ensure the name has an image extension, add .png if missing (adjust if needed)
                if '.' not in name:
                    name += ".png"

            except KeyError as e:
                 logger.error(f"CRITICAL ERROR: Missing key {e} in input data for image index {i}. Cannot determine filename or dimensions. Skipping.")
                 # No metrics lists to append to
                 results_log.append({'img_name': f'error_missing_key_{i}', 'pred_masks': []})
                 continue
            except Exception as e:
                 logger.error(f"CRITICAL ERROR processing input info for image index {i}: {e}. Skipping.", exc_info=True)
                 # No metrics lists to append to
                 results_log.append({'img_name': f'error_processing_input_{i}', 'pred_masks': []})
                 continue

            # --- Run Model Inference ---
            try:
                model_output_list = model(model_input) # Expecting list output
                if not isinstance(model_output_list, list) or not model_output_list:
                     raise ValueError("Model output is not a non-empty list.")
                # Access the dictionary inside the list
                preds_dict = model_output_list[0]
            except Exception as e:
                 logger.error(f"ERROR during model forward pass for {name}: {e}. Saving blank map.", exc_info=True)
                 # No metrics lists to append to
                 results_log.append({'img_name': name, 'pred_masks': []})
                 # Save a blank map on error
                 pred_map = np.zeros((h, w), dtype=np.uint8) # Use extracted h, w
                 save_path = os.path.join(sal_map_out, name)
                 cv2.imwrite(save_path, pred_map)
                 continue # Skip to next image

            # --- Process Predictions using 'instances' key ---
            pred_map = np.zeros((h, w), dtype=np.uint8) # Default to blank map
            kept_masks_np = np.zeros([0, h, w], dtype=np.float32) # Default to empty masks

            if "instances" in preds_dict:
                try:
                    instances = preds_dict["instances"].to("cpu") # Get Instances object
                    pred_saliency_filtered = Instances(instances.image_size) # Create empty Instances
                    flag = False # Flag to track if any instances pass threshold

                    # Iterate through detected instances
                    for idx in range(len(instances)):
                        # Access scores from the Instances object
                        score = instances.scores[idx].item()
                        if score > confidence_threshold:
                            if not flag:
                                pred_saliency_filtered = instances[idx] # Assign first passing instance
                                flag = True
                            else:
                                # Concatenate subsequent passing instances
                                pred_saliency_filtered = Instances.cat([pred_saliency_filtered, instances[idx]])

                    # --- Generate Final Prediction Map ---
                    if flag: # If at least one instance passed
                        # Access masks from the filtered Instances object
                        pred_masks = pred_saliency_filtered.pred_masks.numpy() # Get numpy masks [N_kept, H, W]
                        # Combine kept masks (union)
                        pred_map = (np.any(pred_masks, axis=0).astype(np.uint8)) * 255
                        kept_masks_np = pred_masks
                        # logger.info(f"Image: {name} ------ Kept {len(pred_saliency_filtered)} instances.")
                    # else: # No instances passed threshold, pred_map remains blank
                        # logger.warning(f"Image: {name} ------ No predicted instances passed threshold {confidence_threshold}")

                    # Append results for logging (only predicted masks)
                    results_log.append({
                        'img_name': name,
                        # 'gt_masks': [], # GT masks not needed if not evaluating
                        'pred_masks': [mask for mask in kept_masks_np]
                    })

                except (AttributeError, KeyError, Exception) as e:
                     logger.error(f"ERROR processing 'instances' object for {name}: {e}. Saving blank map.", exc_info=True)
                     # No metrics lists to append to
                     results_log.append({'img_name': name, 'pred_masks': []})
                     # pred_map is already default blank

            else:
                # 'instances' key was not found in the model output dictionary
                logger.warning(f"Image: {name} --- Key 'instances' not found in prediction dictionary. Output keys: {list(preds_dict.keys())}. Saving blank map.")
                # pred_map is already default blank
                # Append results for logging (no predictions)
                results_log.append({
                    'img_name': name,
                    'pred_masks': []
                })
                # No metrics lists to append to

            # --- Save Map --- (Metrics calculation removed)
            save_path = os.path.join(sal_map_out, name)
            try:
                success = cv2.imwrite(save_path, pred_map)
                if not success:
                    logger.error(f"Failed to write image at: {save_path}. Check permissions and path validity.")
            except Exception as e:
                 logger.error(f"Exception during cv2.imwrite for {save_path}: {e}", exc_info=True)

    return results_log


# --- Inference Function ---
def inference(cfg, model, split):
    logger.info(f"Setting up dataset '{cfg.DATASETS.NAME}' for split '{split}'...")

    current_dataset_dir = DATASET_DIR 
    logger.info(f"Using dataset root directory: {current_dataset_dir}")

    if cfg.DATASETS.NAME == "hicost":
        try:
            mapper = HicostDatasetMapper(cfg=cfg, is_train=False)
        except TypeError:
             logger.warning("HicostDatasetMapper might not accept cfg directly in __init__. Trying without.")
             mapper = HicostDatasetMapper(is_train=False)

        dataset_dicts = get_dict(root=current_dataset_dir, split=split)
        if not dataset_dicts:
             logger.error(f"Dataset dictionary for split '{split}' is empty or None. Check dataset path '{current_dataset_dir}' and JSON file structure/content.")
             return []

        # Verify dataset_dicts structure
        if not isinstance(dataset_dicts, list) or not all(isinstance(item, dict) for item in dataset_dicts):
            logger.error(f"Expected dataset_dicts to be a list of dictionaries, but got {type(dataset_dicts)}. Check get_dict function.")
            return []

        # Check if dataset is empty after loading
        if not dataset_dicts:
             logger.error(f"Loaded dataset for split '{split}' is empty. Check dataset files.")
             return []


        dataset_lst = DatasetFromList(dataset_dicts, copy=False)
        dataset_mapped = MapDataset(dataset_lst, mapper) # Apply mapper
        # Use cfg.SOLVER.IMS_PER_BATCH for batch size during inference? Usually 1 for eval.
        dataloader = DataLoader(dataset_mapped, batch_size=1, collate_fn=trivial_batch_collator, num_workers=cfg.DATALOADER.NUM_WORKERS)
        logger.info(f"DataLoader created with batch size 1 and {cfg.DATALOADER.NUM_WORKERS} workers.")
    else:
        # Handle other datasets if necessary
        raise ValueError(f"Unknown or unsupported dataset configuration name: {cfg.DATASETS.NAME}")

    # --- Output Path Calculation ---
    base_output_dir = cfg.EVALUATION.OUTPUT_DIR
    if not base_output_dir:
        logger.error("cfg.EVALUATION.OUTPUT_DIR is not set! Cannot determine where to save results.")
        # Decide how to handle: maybe default or raise error
        base_output_dir = "./evals/results" # Fallback default
        logger.warning(f"cfg.EVALUATION.OUTPUT_DIR was not set. Defaulting to {base_output_dir}")

    model_weights_path = cfg.MODEL.WEIGHTS
    if not model_weights_path:
        logger.warning("cfg.MODEL.WEIGHTS is not set. Using 'unknown_model' for output subdirectory name.")
        model_name_for_path = "unknown_model"
    else:
        # Create a safe directory name from the weights filename
        model_basename = os.path.basename(model_weights_path)
        model_name_for_path = os.path.splitext(model_basename)[0]

    output_subdir = os.path.join(base_output_dir, f"{cfg.DATASETS.NAME}_{split}", model_name_for_path)
    sal_map_out = os.path.join(output_subdir, "saliency_maps")
    print(f"Output directory for saliency maps: {sal_map_out}")

    # Create directory only on the main process to avoid race conditions
    if comm.is_main_process():
        try:
            if not os.path.exists(sal_map_out):
                logger.info(f"Creating output directory: {sal_map_out}")
                os.makedirs(sal_map_out, exist_ok=True) # Use exist_ok=True
        except OSError as e:
            logger.error(f"Error creating output directory {sal_map_out}: {e}")
            sys.exit(1)

    comm.synchronize()

    logger.info("Starting inference loop (saving maps only)...")
    # Pass cfg to run_inference_loop as it's needed for threshold/weights info
    results = run_inference_loop(dataloader, model, cfg, sal_map_out)
    logger.info("Inference loop finished.")

    return results

# --- Main Guard ---
if __name__ == "__main__":
    print("inference.py should be imported by a main script (like test.py), not run directly.")
    pass
