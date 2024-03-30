import numpy as np
import torch
import torchvision
import cv2
import os
import json

from config import *

# (original_image_id + adv_method) for one log file. 
class OutputLogger():
    def __init__(self, log_folder, imgid, adv_method):
        os.makedirs(log_folder, exist_ok=True)
        self.record_filename = f"{log_folder}/{imgid}_{adv_method}.json"
        if os.path.exists(self.record_filename):
            with open(self.record_filename, 'r') as f:
                self.record = json.load(f)
        else:
            self.record = {}
            self.record['imgid'] = imgid
            self.record['adv_method'] = adv_method
            self.record['evaluations'] = {}
        self.evaluations = self.record['evaluations']

    def add_evaluation(self, prompt, evaluator_name, iou):
        if not isinstance(prompt, list):
            prompt = prompt.tolist()
        if prompt not in self.evaluations:
            self.evaluations[prompt] = {}
        self.evaluations[prompt][evaluator_name] = iou

    def add_single_evaluator_result(self, evaluator_name, prompt_list, iou_list):
        mask_num = len(prompt_list)
        for mask_id in range(mask_num):
            for prompt_id, (prompt, iou) in enumerate(zip(prompt_list[mask_id], iou_list[mask_id])):
                prompt_uid = f"m{mask_id}_p{prompt_id}"
                if prompt_uid not in self.evaluations:
                    self.evaluations[prompt_uid] = {}
                    self.evaluations[prompt_uid]['prompt'] = prompt.tolist()
                    self.evaluations[prompt_uid][evaluator_name] = iou
                else:
                    self.evaluations[prompt_uid][evaluator_name] = iou

    def dump_evaluations(self):
        with open(self.record_filename, 'w') as f:
            json.dump(self.record, f, indent=4)
    


def IoU(mask1, mask2):
    if mask1.__class__ == torch.Tensor:
        mask1 = mask1.detach().cpu().numpy()
    if mask2.__class__ == torch.Tensor:
        mask2 = mask2.detach().cpu().numpy()
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    return 100 * np.sum(intersection) / np.sum(union)

def sample_pixel_in_mask(mask, sample_num=10):
    # mask is a numpy 2d-array of shape [H, W]
    # output P point prompts in a list of shape [P, 2]
    inmask_pixel_positions = np.flip(np.argwhere(mask == True), axis=1)
    sample_size = min(sample_num, inmask_pixel_positions.shape[0])
    sampled_pixel_id = np.random.choice(inmask_pixel_positions.shape[0], size=sample_size, replace=False)
    sampled_pixel_pos = inmask_pixel_positions[sampled_pixel_id]
    return sampled_pixel_pos

def save_image(img_data, path):
    if isinstance(img_data, torch.Tensor):
        torchvision.utils.save_image(img_data, path)
    elif isinstance(img_data, np.ndarray):
        cv2.imwrite(path, cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR))
