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

from torchmetrics.detection.mean_ap import MeanAveragePrecision

from minigpt4.datasets.datasets.vqa_datasets import RefADEvalData
from minigpt4.common.vqa_tools.VQA.PythonHelperTools.vqaTools.vqa import VQA
from minigpt4.common.vqa_tools.VQA.PythonEvaluationTools.vqaEvaluation.vqaEval import (
    VQAEval,
)

from minigpt4.common.eval_utils import (
    prepare_texts,
    init_model,
    eval_parser,
    computeIoU,
)
from minigpt4.conversation.conversation import CONV_VISION_minigptv2
from minigpt4.common.config import Config


def list_of_str(arg):
    return list(map(str, arg.split(",")))


parser = eval_parser()
parser.add_argument(
    "--dataset", type=list_of_str, default="refcoco", help="dataset to evaluate"
)
parser.add_argument(
    "--resample", action="store_true", default=False, help="resample failed samples"
)
parser.add_argument(
    "--res", type=float, default=100.0, help="resolution used in regresion"
)
args = parser.parse_args()
cfg = Config(args)

model, vis_processor = init_model(args)
conv_temp = CONV_VISION_minigptv2.copy()
conv_temp.system = ""
model.eval()
save_path = cfg.run_cfg.save_path

if "mvtech" in args.dataset:
    eval_file_path = cfg.evaluation_datasets_cfg["mvtech_ad"]["eval_file_path"]
    batch_size = cfg.evaluation_datasets_cfg["mvtech_ad"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["mvtech_ad"]["max_new_tokens"]

    # Adapt the data loading to the RefCOCO format
    with open(eval_file_path, "r") as f:
        mvtech_ad_data_for_regression = json.load(f)

    # Limit the data to 10% of the original data
    mvtech_ad_data_for_regression = mvtech_ad_data_for_regression[:len(mvtech_ad_data_for_regression)//10]
    # mvtech_ad_data_for_regression = mvtech_ad_data_for_regression[:10]
    data = RefADEvalData(mvtech_ad_data_for_regression, vis_processor)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)

    minigpt4_predict = defaultdict(list)
    resamples = []

    for images, questions, img_ids, labels, image_paths in tqdm(eval_dataloader):
        texts = prepare_texts(questions, conv_temp)
        answers = model.generate(
            images, texts, max_new_tokens=max_new_tokens, do_sample=False
        )
        for answer, img_id, question, labels, image_paths in zip(
            answers, img_ids, questions, labels, image_paths
        ):
            answer = answer.replace("<unk>", "").replace(" ", "").strip()
            pattern = r"<p>(.*?)<\/p>\{<\d{1,3}><\d{1,3}><\d{1,3}><\d{1,3}>\}"
            if re.match(pattern, answer):
                minigpt4_predict[img_id].append(answer)
            else:
                resamples.append(
                    {"img_id": img_id, "class": labels, "img_path": image_paths}
                )

    if args.resample:
        for i in range(20):
            data = RefADEvalData(resamples, vis_processor)
            resamples = []
            eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)

            for images, questions, img_ids, labels, image_paths in tqdm(
                eval_dataloader
            ):
                texts = prepare_texts(questions, conv_temp)
                answers = model.generate(
                    images, texts, max_new_tokens=max_new_tokens, do_sample=False
                )
                for answer, img_id, question, labels, image_paths in zip(
                    answers, img_ids, questions, labels, image_paths
                ):
                    answer = answer.replace("<unk>", "").replace(" ", "").strip()
                    pattern = r"<p>(.*?)<\/p>\{<\d{1,3}><\d{1,3}><\d{1,3}><\d{1,3}>\}"
                    if re.match(pattern, answer) or i == 4:
                        minigpt4_predict[img_id].append(answer)
                    else:
                        resamples.append(
                            {"img_id": img_id, "class": labels, "img_path": image_paths}
                        )

            if len(resamples) == 0:
                break

    # Save predictions
    file_save_path = os.path.join(save_path, "mvtech_ad_regression.json")
    with open(file_save_path, "w") as f:
        json.dump(minigpt4_predict, f)

    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)

    # Calculate metrics
    count = 0
    total = 0
    res = args.res
    for item in mvtech_ad_data_for_regression:
        img_id = (
            item["class"] + "." + os.path.basename(item["image_path"]).split(".")[0]
        )
        label = item["class"]
        is_broken = item["is_broken"]
        outputs = minigpt4_predict[img_id]

        # Determine ground truth bounding box and class
        if not is_broken:
            # If not broken, the bounding box is the whole image
            gt_bbox = [0, 0, item["width"], item["height"]]
            gt_class = 0  # Class 0 for "not-defect"
        else:
            gt_bbox = [
                item["bbox"][0],
                item["bbox"][1],
                item["bbox"][2],
                item["bbox"][3],
            ]
            gt_class = 1  # Class 1 for "defect"

        # Ground truth data for torchmetrics
        gt_boxes = torch.tensor([gt_bbox], dtype=torch.float)
        gt_labels = torch.tensor([gt_class], dtype=torch.int)

        outputs = minigpt4_predict[img_id]

        pred_boxes = []
        pred_scores = []
        pred_labels = []

        for output in outputs:
            match = re.search(
                r"<p>(.*?)<\/p>\{<(\d{1,3})><(\d{1,3})><(\d{1,3})><(\d{1,3})>\}",
                output,
            )
            if match:
                try:
                    pred_class_str = match.group(1).strip()

                    # Determine predicted class based on the string
                    if "not-defect" in pred_class_str:
                        pred_class = 0
                    elif "defect" in pred_class_str:
                        pred_class = 1
                    else:
                        pred_class = -1  # Or some other value to indicate uncertain

                    pred_bbox = [
                        int(match.group(2)),
                        int(match.group(3)),
                        int(match.group(4)),
                        int(match.group(5)),
                    ]
                    height = item["height"]
                    width = item["width"]
                    pred_bbox[0] = pred_bbox[0] / res * width
                    pred_bbox[1] = pred_bbox[1] / res * height
                    pred_bbox[2] = pred_bbox[2] / res * width
                    pred_bbox[3] = pred_bbox[3] / res * height

                    pred_boxes.append(pred_bbox)
                    pred_scores.append(1.0)  # Assuming confidence score of 1
                    pred_labels.append(pred_class)
                except Exception as e:
                    print(f"Error processing output: {output}, Error: {e}")
                    continue

        # Convert lists to tensors
        if pred_boxes:
            pred_boxes = torch.tensor(pred_boxes, dtype=torch.float)
            pred_scores = torch.tensor(pred_scores, dtype=torch.float)
            pred_labels = torch.tensor(pred_labels, dtype=torch.int)
        else:
            # Create empty tensors if no predictions were made
            pred_boxes = torch.empty((0, 4), dtype=torch.float)
            pred_scores = torch.empty(0, dtype=torch.float)
            pred_labels = torch.empty(0, dtype=torch.int)

        # Update metric
        metric.update(
            [dict(boxes=pred_boxes, scores=pred_scores, labels=pred_labels)],
            [dict(boxes=gt_boxes, labels=gt_labels)],
        )

    # Compute metric
    result = metric.compute()
    map_value = result["map"].item()

    # Print class-wise metrics
    for i, class_map in enumerate(result["map_per_class"]):
        class_name = "defect" if i == 1 else "not-defect"
        print(
            f"mAP for {class_name}: {class_map.item() * 100 if not torch.isnan(class_map) else 0:.4f}",
            flush=True,
        )
    print(f"mAP: {map_value * 100}", flush=True)
