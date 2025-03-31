import os
import re
import json
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.utils.data import DataLoader
from minigpt4.common.config import Config
from minigpt4.common.eval_utils import prepare_texts, init_model, eval_parser
from minigpt4.conversation.conversation import CONV_VISION_minigptv2

def list_of_str(arg):
    return list(map(str, arg.split(',')))

parser = eval_parser()
parser.add_argument("--dataset", type=list_of_str, default='refcoco', help="dataset to evaluate")
parser.add_argument("--res", type=float, default=100.0, help="resolution used in refcoco")
parser.add_argument("--resample", action='store_true', help="resolution used in refcoco")
args = parser.parse_args()

cfg = Config(args)

model, vis_processor = init_model(args)
model.eval()
CONV_VISION = CONV_VISION_minigptv2
conv_temp = CONV_VISION.copy()
conv_temp.system = ""
save_path = cfg.run_cfg.save_path

os.makedirs(save_path, exist_ok=True)


class RefADEvalData(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor):
        self.loaded_data = loaded_data
        self.vis_processor = vis_processor

    def __len__(self):
        return len(self.loaded_data)

    def __getitem__(self, idx):
        data = self.loaded_data[idx]
        sent = "[detection] a defect or not-defect object and return the bounding boxes and its label. If not, bound around the object."
        tag = "good" if not data["is_broken"] else "broken"
        img_id = (data["class"] + "." + tag + "."  + os.path.basename(data["image_path"]).split(".")[0])
        fix_path = os.path.join("./data/MVTEC_det/images", "/".join(data["image_path"].split("/")[1:4]))
        image = Image.open(fix_path).convert("RGB")
        image = self.vis_processor(image)
        return image, sent, img_id, data["class"], data["image_path"]


eval_file_path = cfg.evaluation_datasets_cfg["mvtec_ad"]["eval_file_path"]
batch_size = cfg.evaluation_datasets_cfg["mvtec_ad"]["batch_size"]
max_new_tokens = cfg.evaluation_datasets_cfg["mvtec_ad"]["max_new_tokens"]


# Adapt the data loading to the RefCOCO format
with open(eval_file_path, "r") as f:
    mvtec_ad_data_for_regression = json.load(f)

data = RefADEvalData(mvtec_ad_data_for_regression, vis_processor)
eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)

minigpt4_predict = defaultdict(list)
resamples = []

for images, questions, img_ids, labels, image_paths in tqdm(eval_dataloader):
    texts = prepare_texts(questions, conv_temp)
    answers = model.generate(images, 
                             texts, 
                             max_new_tokens=max_new_tokens, 
                             do_sample=False)
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

        for images, questions, img_ids, labels, image_paths in tqdm(eval_dataloader):
            texts = prepare_texts(questions, conv_temp)
            answers = model.generate(images, 
                                     texts, 
                                     max_new_tokens=max_new_tokens, 
                                     do_sample=False)
            for answer, img_id, question, labels, image_paths in zip(
                answers, img_ids, questions, labels, image_paths
            ):
                answer = answer.replace("<unk>", "").replace(" ", "").strip()
                pattern = r"<p>(.*?)<\/p>\{<\d{1,3}><\d{1,3}><\d{1,3}><\d{1,3}>\}"
                if re.match(pattern, answer) or i == 4:
                    minigpt4_predict[img_id].append(answer)
                else:
                    resamples.append({
                        "img_id": img_id, 
                        "class": labels, 
                        "img_path": image_paths
                    })
        if len(resamples) == 0:
            break


# Save predictions
file_save_path = os.path.join(save_path, "mvtec_ad_regression.json")
with open(file_save_path, "w") as f:
    json.dump(minigpt4_predict, f, indent=4)


metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)

# Calculate metrics
count = 0
total = 0
res = args.res
for item in mvtec_ad_data_for_regression:
    is_broken = item["is_broken"]
    label = item["class"]
    tag = "good" if not item["is_broken"] else "broken"
    img_id = (item["class"] + "." + tag + "."  + os.path.basename(item["image_path"]).split(".")[0])
    if img_id not in minigpt4_predict:
        continue
    outputs = minigpt4_predict[img_id]

    # Determine ground truth bounding box and class
    if not is_broken:
        # If not broken, the bounding box is the whole image
        gt_bbox = [0, 0, 100, 100]
        gt_class = 0  # Class 0 for "not-defect"
    else:
        gt_bbox = [
            item["bbox"][0],
            item["bbox"][1],
            item["bbox"][2],
            item["bbox"][3]
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
                pred_bbox[0] = pred_bbox[0]
                pred_bbox[1] = pred_bbox[1]
                pred_bbox[2] = pred_bbox[2]
                pred_bbox[3] = pred_bbox[3]


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

print(f"mAP: {map_value * 100}", flush=True)
