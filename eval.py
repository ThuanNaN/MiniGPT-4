import os
import re
import json
import argparse
import cv2
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

from minigpt4.datasets.datasets.vqa_datasets import RefADEvalData
from minigpt4.common.vqa_tools.VQA.PythonHelperTools.vqaTools.vqa import VQA
from minigpt4.common.vqa_tools.VQA.PythonEvaluationTools.vqaEvaluation.vqaEval import VQAEval

from minigpt4.common.eval_utils import prepare_texts, init_model, eval_parser, computeIoU
from minigpt4.conversation.conversation import CONV_VISION_minigptv2
from minigpt4.common.config import Config


def list_of_str(arg):
    return list(map(str, arg.split(',')))

parser = eval_parser()
parser.add_argument("--dataset", type=list_of_str, default='refcoco', help="dataset to evaluate")
parser.add_argument("--resample", action="store_true", default=False, help="resample failed samples")
parser.add_argument("--res", type=float, default=100.0, help="resolution used in regresion")
args = parser.parse_args()
cfg = Config(args)

model, vis_processor = init_model(args)
conv_temp = CONV_VISION_minigptv2.copy()
conv_temp.system = ""
model.eval()
save_path = cfg.run_cfg.save_path

if 'mvtech' in args.dataset:
    eval_file_path = cfg.evaluation_datasets_cfg["mvtech_ad"]["eval_file_path"]
    batch_size = cfg.evaluation_datasets_cfg["mvtech_ad"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["mvtech_ad"]["max_new_tokens"]
    
    mvtech_ad = []
    
    # Adapt the data loading to the RefCOCO format
    mvtech_ad_data_for_regression = []
    for category in os.listdir(eval_file_path):
        category_path = os.path.join(eval_file_path, category)
        if os.path.isdir(category_path):
            for split in ["test"]: 
                split_path = os.path.join(category_path, split)
                if os.path.isdir(split_path):
                    for defect in os.listdir(split_path):
                        defect_path = os.path.join(split_path, defect)
                        for img_file in os.listdir(defect_path):
                            img_path = os.path.join(defect_path, img_file)
                            mask_path = os.path.join(category_path, "ground_truth", defect, img_file.replace(".png", "_mask.png"))

                            if os.path.exists(mask_path):
                                # Get bounding box from mask
                                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                if contours:
                                    x, y, w, h = cv2.boundingRect(contours[0])
                                    bbox = [x, y, x + w, y + h]

                                    img_id = f"{category}_{split}_{defect}_{img_file}" 
                                    
                                    mvtech_ad_data_for_regression.append({
                                        "img_id": img_id,
                                        "img_path": img_path,
                                        "category": category,
                                        "defect": defect,
                                        "bbox": bbox,
                                        "height": mask.shape[0],  # Assuming mask dimensions match image
                                        "width": mask.shape[1],
                                        "sents": '[refer] give me the location of ' + defect.replace('_', ' ') + ' defect'
                                    })
                            
    data = RefADEvalData(mvtech_ad_data_for_regression, vis_processor)
    data = list(data)[:len(data)//10]  # Limit to 10% of the data
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)

    minigpt4_predict = defaultdict(list)
    resamples = []

    for images, questions, img_ids in tqdm(eval_dataloader):
        texts = prepare_texts(questions, conv_temp)
        answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)
        for answer, img_id, question in zip(answers, img_ids, questions):
            answer = answer.replace("<unk>","").replace(" ","").strip()
            pattern = r'\{<\d{1,3}><\d{1,3}><\d{1,3}><\d{1,3}>\}'
            if re.match(pattern, answer):
                minigpt4_predict[img_id].append(answer)
            else:
                resamples.append({'img_id': img_id, 'sents': question.replace('[refer] give me the location of','').strip()})

    if args.resample:
        for i in range(20):
            data = RefADEvalData(resamples, vis_processor)
            resamples = []
            eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)

            for images, questions, img_ids in tqdm(eval_dataloader):
                texts = prepare_texts(questions, conv_temp)
                answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)
                for answer, img_id, question in zip(answers, img_ids, questions):
                    answer = answer.replace("<unk>","").replace(" ","").strip()
                    pattern = r'\{<\d{1,3}><\d{1,3}><\d{1,3}><\d{1,3}>\}'
                    if re.match(pattern, answer) or i == 4:
                        minigpt4_predict[img_id].append(answer)
                    else:
                        resamples.append({'img_id': img_id, 'sents': question.replace('[refer] give me the location of','').strip()})

            if len(resamples) == 0:
                break

    # Save predictions
    file_save_path = os.path.join(save_path, "mvtech_ad_regression.json")
    with open(file_save_path, 'w') as f:
        json.dump(minigpt4_predict, f)

    # Calculate metrics
    count = 0
    total = len(mvtech_ad_data_for_regression)
    res = args.res
    for item in mvtech_ad_data_for_regression:
        img_id = item['img_id']
        bbox = item['bbox']
        outputs = minigpt4_predict[img_id]

        for output in outputs:
            try:
                integers = re.findall(r'\d+', output)
                pred_bbox = [int(num) for num in integers]
                height = item['height']
                width = item['width']
                pred_bbox[0] = pred_bbox[0] / res * width
                pred_bbox[1] = pred_bbox[1] / res * height
                pred_bbox[2] = pred_bbox[2] / res * width
                pred_bbox[3] = pred_bbox[3] / res * height

                gt_bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]

                iou_score = computeIoU(pred_bbox, gt_bbox)
                if iou_score > 0.5:
                    count += 1
            except:
                continue

    print(f'MVTech AD (Regression):', count / total * 100, flush=True)