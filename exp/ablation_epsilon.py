import torch
import torch.nn as nn

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from attack.eval import *
from attack import SamAttacker
from utils import *
from config import *
from segment_anything import sam_model_registry, SamPredictor
from FastSAM.fastsam import FastSAM, FastSAMPrompt

def set_attacker(device):
    model_type = "vit_b"
    samb_model = sam_model_registry[model_type](checkpoint=SAM_B_PATH)
    samb_model.to(device=device)
    attacker = SamAttacker(samb_model)
    return attacker

def set_evaluators(evaluator_list, device_list):
    assert len(evaluator_list) == len(device_list)
    evaluator_dict = {}
    for evaluator, device in zip(evaluator_list, device_list):
        if evaluator == 'samb':
            samb_model = sam_model_registry['vit_b'](checkpoint=SAM_B_PATH).to(device)
            samb_eval = SamPredictor(samb_model)
            evaluator_dict[evaluator] = samb_eval
        elif evaluator == 'saml':
            saml_model = sam_model_registry['vit_l'](checkpoint=SAM_L_PATH).to(device)
            saml_eval = SamPredictor(saml_model)
            evaluator_dict[evaluator] = saml_eval
        elif evaluator == 'samh':
            samh_model = sam_model_registry['vit_h'](checkpoint=SAM_H_PATH).to(device)
            samh_eval = SamPredictor(samh_model)
            evaluator_dict[evaluator] = samh_eval
        elif evaluator == 'fastsam':
            fastsam = FastSAM(FASTSAM_PATH)
            fastsam.model.to(device)
            evaluator_dict[evaluator] = fastsam
        else:
            raise ValueError(f"Unknown evaluator {evaluator}, select from ['samb', 'saml', 'samh', 'fastsam']")
    return evaluator_dict

def single_evaluator_job(adv_img_cv2, evaluator, gt_mask_list, prompt_list):
    if isinstance(evaluator, SamPredictor):
        adv_img_torch = evaluator.set_image(adv_img_cv2)
        iou_list = sam_single_image_batched_prompt_test(evaluator, gt_mask_list, prompt_list)
    elif isinstance(evaluator, FastSAM):
        iou_list = fastsam_single_image_batched_prompt_test(evaluator, adv_img_cv2, gt_mask_list, prompt_list)
    else:
        raise ValueError(f"Unknown evaluator {evaluator}")
    
    return iou_list


def evaluate_imgid_attackname(adv_img_cv2, imgid, attackname, evaluator_dict, log_folder, parallel=False):
    logger = OutputLogger(log_folder=log_folder, imgid=imgid, adv_method=attackname)
    imgpath = f"{SAM_DATASET_PATH}sa_{imgid}.jpg"
    gt_mask_list, prompt_list = obtain_single_image_evaluation_config(imgpath)
    
    if parallel == False: # sequential evaluation
        for evaluator_name, evaluator in evaluator_dict.items():
            single_evaluator_iou_list = single_evaluator_job(adv_img_cv2, evaluator, gt_mask_list, prompt_list)
            logger.add_single_evaluator_result(evaluator_name, prompt_list, single_evaluator_iou_list)
    else: # parallel evaluation
        pass

    logger.dump_evaluations()

if __name__ == '__main__':
    evaluator_list = ['fastsam', 'samb', 'saml', 'samh']
    device_list = ['cuda:0', 'cuda:5', 'cuda:5', 'cuda:6']
    attacker_device = 'cuda:4'
    attack_namelist = ['attack_sam_k', 'AA', 'PATA', 'PATA++', 'UAD']
    epsilon_list = [4, 8, 12, 16]
    evaluator_dict = set_evaluators(evaluator_list, device_list)
    attacker = set_attacker(device=attacker_device)
    for epsilon in epsilon_list:
        attacker.adv_epsilon = float(epsilon/255)
        for attack_name in attack_namelist:
            for imgid in range(1, 41):
                log_folder = f"{PROJECT_PATH}/exp/log/epsilon/{attack_name}_eps_{epsilon}/"
                if os.path.exists(f"{log_folder}{imgid}_{attack_name}.json"):
                    continue
                else:
                    print(f"Processing img {imgid} with {attack_name} at epsilon {epsilon}") 
                try:
                    adv_input = attacker.attack_fn(attack_name.lower())(f"{SAM_DATASET_PATH}sa_{imgid}.jpg")
                    adv_cv2_img = attacker.get_cv2_from_torch(adv_input)
                    evaluate_imgid_attackname(adv_cv2_img, imgid, attack_name, evaluator_dict, log_folder)
                except Exception as e:
                    print(f"Numerical instablity error in processing {imgid} with {attack_name}: {e}")
                    continue