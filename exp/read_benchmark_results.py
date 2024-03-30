import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *
from config import *


attack_name_list = ['TAP', 'ILPD', 'AA', 'PATA', 'PATA++', 'UAD']
for attack_name in attack_name_list:
    evaluator_list = ['fastsam', 'samb', 'saml', 'samh']
    log_folder = f"{PROJECT_PATH}/exp/log/benchmark/{attack_name}/"
    iou_evaluator_list = [[], [], [], []]
    for imgid in range(1, 1001):
        result_json_file_name = f"{log_folder}{imgid}_{attack_name}.json"
        if imgid % 100 == 0:
                print(f"Processed {imgid} images")
        if os.path.exists(result_json_file_name):
            with open(result_json_file_name, 'r') as f:
                results = json.load(f)
                evaluations = results['evaluations']
                for prompt_uid, prompt_result in evaluations.items():
                    for evaluator_id, evaluator in enumerate(evaluator_list):
                        iou = prompt_result[evaluator]
                        iou_evaluator_list[evaluator_id].append(iou)
        else:
            print(f"File {result_json_file_name} does not exist")
            continue
        
    print(f"Evaluation results for {attack_name}:")
    for iou_evaluator, evaluator in zip(iou_evaluator_list, evaluator_list):
        iou_mean = np.mean(iou_evaluator)
        iou_std = np.std(iou_evaluator)
        asr_50 = np.sum(np.array(iou_evaluator) <= 50) / len(iou_evaluator) * 100
        asr_10 = np.sum(np.array(iou_evaluator) <= 10) / len(iou_evaluator) * 100
        print(f"Evaluator {evaluator} - {iou_mean:.2f}+-{iou_std:.2f} - ASR@50: {asr_50:.2f} - ASR@10: {asr_10:.2f}")