import os
import json
from collections import Counter
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
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


class EvalTextVQAData(torch.utils.data.Dataset):
    def __init__(self, loaded_data, image_processor):
        self.loaded_data = loaded_data
        self.image_processor = image_processor

    def __len__(self):
        return len(self.loaded_data)

    def __getitem__(self, idx):
        data = self.loaded_data[idx]
        question = data["question"]
        question = f"[vqa] {question}"
        image = Image.open(data["image_path"]).convert("RGB")
        image = self.image_processor(image)
        return image, question, data["image_id"], data["answers"]


eval_file_path = cfg.evaluation_datasets_cfg["textvqa"]["eval_file_path"]
img_path = cfg.evaluation_datasets_cfg["textvqa"]["img_path"]
batch_size = cfg.evaluation_datasets_cfg["textvqa"]["batch_size"]
max_new_tokens = cfg.evaluation_datasets_cfg["textvqa"]["max_new_tokens"]

with open(eval_file_path, "r") as f:
    train_data = json.load(f)

data = []
for item in train_data:
    data.append(
        {
            "question": item["question"],
            "image_id": item["image_id"],
            "image_path": os.path.join(img_path, item["image_id"] + ".jpg"),
            "answers": item["answers"],
        }
    )

textvqa = EvalTextVQAData(data, vis_processor)
eval_dataloader = DataLoader(textvqa, batch_size=batch_size, shuffle=False)

count = 0
total = 0
minigpt4_predict = []
print("Evaluating on TextVQA dataset")
for images, texts, image_id, labels in tqdm(eval_dataloader):
    texts = prepare_texts(
        texts, conv_temp
    )  # warp the texts with conversation template
    answers = model.generate(images,
                             texts, 
                             max_new_tokens=max_new_tokens, 
                             do_sample=False)

    # Stack the labels to correct order (transpose)
    labels = [list(x) for x in zip(*labels)]

    for answer, labels in zip(answers, labels):
        result = dict()
        result["pred"] = answer.lower().replace("<unk>", "").strip()
        result["gt"] = Counter(labels).most_common(1)[0][0]

        minigpt4_predict.append(result)
        if answer.lower() == result["gt"]:
            count += 1
        total += 1

        # Calculate BLEU score
        chencherry = SmoothingFunction()
        bleu_score = sentence_bleu(
            labels, answer, smoothing_function=chencherry.method1
        )
        result["bleu"] = bleu_score


print("Saving predictions to", save_path)
file_save_path = os.path.join(save_path, "textvqa.json")
with open(file_save_path, "w") as f:
    json.dump(minigpt4_predict, f, indent=4)

print("Top 1 Accuracy:", count / total * 100, flush=True)
print("Average BLEU score: ",
       np.mean([pred["bleu"] for pred in minigpt4_predict]),
       flush=True)
