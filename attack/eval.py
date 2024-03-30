import numpy as np
import torch
import cv2
import json
from pycocotools.mask import decode as pycocotools_decode

from segment_anything import SamPredictor
from FastSAM.fastsam import FastSAM, FastSAMPrompt
from attack import SamAttacker
from utils import *


def obtain_single_image_evaluation_config(
    imgpath: str,
    eval_point_sample_num: int=5,
    eval_box_sample_num: int=3,
    eval_box_sample_rescale_low: float=0.9,
    eval_box_sample_rescale_high: float=1.1,
    mask_num_limit: int=10
):
    #clean_cv2_image = cv2.imread(imgpath)
    #clean_cv2_image = cv2.cvtColor(clean_cv2_image, cv2.COLOR_BGR2RGB)
    mask_list = [] # list of mask: M-sequence of [H, W]-ndarray
    prompt_list = [] # list of list of prompt: M lists, each list containing (P+B)-sequence of [2 or 4]-ndarray
    labelpath = imgpath.replace('.jpg', '.json')
    with open(labelpath, 'r') as f:
        label_data = json.load(f)
        gt_masks = label_data['annotations']
        gt_masks = gt_masks[:mask_num_limit]
        for mask_id, gt_mask in enumerate(gt_masks): # from 0 to M-1
            gt_segmentation = pycocotools_decode(gt_mask['segmentation'])
            gt_height = gt_mask['segmentation']['size'][0]
            gt_width = gt_mask['segmentation']['size'][1]
            gt_bbox = gt_mask['bbox']
            prompt_list_for_current_mask = []
            
            mask_list.append(gt_segmentation)

            # sample P point prompts per mask
            points = sample_pixel_in_mask(gt_segmentation, sample_num=eval_point_sample_num)
            for point_prompt in points:
                prompt_list_for_current_mask.append(point_prompt)
            # sample B box prompts per mask
            cx = gt_bbox [0] + gt_bbox [2] / 2
            cy = gt_bbox [1] + gt_bbox [3] / 2
            w = gt_bbox[2]
            h = gt_bbox[3]
            for box_prompt_id in range(eval_box_sample_num):
                randw = np.random.randint(int(eval_box_sample_rescale_low * w), int(eval_box_sample_rescale_high * w))
                randh = np.random.randint(int(eval_box_sample_rescale_low * h), int(eval_box_sample_rescale_high * h))
                rand_xl = max(int(cx - randw/2), 0)
                rand_xr = min(int(cx + randw/2), gt_width - 1)
                rand_yu = max(int(cy - randh/2), 0)
                rand_yd = min(int(cy + randh/2), gt_height - 1)
                rand_bbox = np.array([rand_xl, rand_yu, rand_xr, rand_yd])
                prompt_list_for_current_mask.append(rand_bbox)
            prompt_list.append(prompt_list_for_current_mask)
    return mask_list, prompt_list


# assume the adversarial input has already been set
@torch.no_grad()
def sam_single_image_batched_prompt_test(
    sam_model: SamPredictor,
    gt_mask_list: list,
    prompt_list: list,
):
    # current implementation only supports single image input + single prompt input
    assert len(gt_mask_list) == len(prompt_list)
    iou_list = []
    for gt_mask_id, (gt_mask, prompt_current_mask_list) in enumerate(zip(gt_mask_list, prompt_list)):
        current_mask_iou_list = []
        #import pdb; pdb.set_trace()
        for prompt_id, prompt in enumerate(prompt_current_mask_list):
            if len(prompt) == 2: # point prompt
                point = np.array([prompt])
                label = np.array([1])
                pred_mask, _, _ = sam_model.predict(point, label, None, multimask_output=False, return_logits=False)
            elif len(prompt) == 4: # box prompt
                box = np.array(prompt)
                pred_mask, _, _ = sam_model.predict(None, None, box, multimask_output=False, return_logits=False)
            pred_mask_iou = IoU(pred_mask[0], gt_mask)
            current_mask_iou_list.append(pred_mask_iou)
        iou_list.append(current_mask_iou_list)
    return iou_list

@torch.no_grad()
def fastsam_single_image_batched_prompt_test(
    fastsam_model: FastSAM,
    adv_img_cv2: np.ndarray,
    gt_mask_list: list,
    prompt_list: list,
):
    iou_list = []
    h, w = adv_img_cv2.shape[:2]
    everything_results = fastsam_model(adv_img_cv2, retina_masks=True, imgsz=max(h, w), conf=0.4, iou=0.9)
    prompt_process = FastSAMPrompt(adv_img_cv2, everything_results)
    for gt_mask_id, (gt_mask, prompt_current_mask_list) in enumerate(zip(gt_mask_list, prompt_list)):
        current_mask_iou_list = []
        for prompt_id, prompt in enumerate(prompt_current_mask_list):
            if len(prompt) == 2:
                point = np.array([prompt])
                label = np.array([1])
                pred_mask = prompt_process.point_prompt(points=point, pointlabel=label)
            elif len(prompt) == 4:
                pred_mask = prompt_process.box_prompt(bbox=prompt.tolist())
            if len(pred_mask) == 0: # no mask predicted
                current_mask_iou_list.append(0.0)
                continue
            pred_mask_iou = IoU(pred_mask[0], gt_mask)
            current_mask_iou_list.append(pred_mask_iou)
        iou_list.append(current_mask_iou_list)
    return iou_list