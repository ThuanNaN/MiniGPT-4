import os
import json
import pandas as pd
from collections import Counter
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from minigpt4.common.eval_utils import (
    prepare_texts,
    init_model,
    eval_parser,
)
from minigpt4.conversation.conversation import CONV_VISION_minigptv2
from minigpt4.common.config import Config


def list_of_str(arg):
    return list(map(str, arg.split(",")))


parser = eval_parser()
parser.add_argument("--dataset", type=list_of_str, default='refcoco', help="dataset to evaluate")
parser.add_argument("--res", type=float, default=100.0, help="resolution used in refcoco")
parser.add_argument("--resample", action='store_true', help="resolution used in refcoco")
args = parser.parse_args()
cfg = Config(args)

model, vis_processor = init_model(args)
conv_temp = CONV_VISION_minigptv2.copy()
conv_temp.system = ""
model.eval()
save_path = cfg.run_cfg.save_path

os.makedirs(save_path, exist_ok=True)


class  EvalViVQADataset(torch.utils.data.Dataset):
    def __init__(self, data, vis_processor, vis_root):
        self.data = data
        self.vis_processor = vis_processor
        self.vis_root = vis_root
        self.img_name_prefix = "COCO_train2014_"
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = self.data.iloc[index]
        image_id = str(sample['img_id']).zfill(12)
        image_path = os.path.join(self.vis_root, f"{image_id}.jpg")
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        base_question = sample["question"]
        question = f"[vqa] Based on the image, respond to this question with a short answer: {base_question}"
        return image, question, sample["answer"]


eval_file_path = cfg.evaluation_datasets_cfg["vivqa"]["eval_file_path"]
img_path = cfg.evaluation_datasets_cfg["vivqa"]["img_path"]
batch_size = cfg.evaluation_datasets_cfg["vivqa"]["batch_size"]
max_new_tokens = cfg.evaluation_datasets_cfg["vivqa"]["max_new_tokens"]


test_data = pd.read_csv(eval_file_path)
vivqa_data = EvalViVQADataset(test_data, vis_processor, img_path)
eval_dataloader = DataLoader(vivqa_data, batch_size=batch_size, shuffle=False)

count = 0
total = 0
minigpt4_predict = []
print("Evaluating on ViVQA dataset")
for images, questions, labels in tqdm(eval_dataloader):
    questions = prepare_texts(
        questions, conv_temp
    )  # warp the texts with conversation template
    answers = model.generate(images,
                             questions, 
                             max_new_tokens=max_new_tokens, 
                             do_sample=False)

    for question, answer, label in zip(questions, answers, labels):
        result = {}
        result["pred"] = answer.lower()
        result["gt"] = Counter(labels).most_common(1)[0][0]
        minigpt4_predict.append(result)

        if answer.lower() == label.lower():
            count += 1
        total += 1

print("Saving predictions to", save_path)
file_save_path = os.path.join(save_path, "vivqa.json")
with open(file_save_path, "w", encoding="utf-8") as f:
    json.dump(minigpt4_predict, f, indent=4, ensure_ascii=False)

print("Top 1 Accuracy:", count / total * 100, flush=True)
