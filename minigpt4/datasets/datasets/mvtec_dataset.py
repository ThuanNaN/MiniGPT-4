import os
import json
import random
from PIL import Image
from torch.utils.data import Dataset


class MVTecDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.instruction_pool = [
            '[detection] {}',
        ]

        with open(ann_path, 'r') as f:
            self.ann = json.load(f)

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        info = self.ann[index]
        gt_bbox = info["bbox"]
        ans_cls = info["class"]

        image_path = os.path.join(self.vis_root, info['image_path'])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        input = "a defect or not-defect object and return the bounding boxes and its label. If not, bound around the object."
        
        ans_defect = "defect" if info["is_broken"] == True else "not-defect"
        ans_para = f"<p>{ans_cls}-{ans_defect}</p>"
        answer = f"{ans_para}{{<{gt_bbox[0]}><{gt_bbox[1]}><{gt_bbox[2]}><{gt_bbox[3]}>}}"

        instruction = random.choice(self.instruction_pool).format(input)
        instruction = "<Img><ImageHere></Img> {} ".format(instruction)

        return {
            "image": image,
            "instruction_input": instruction,
            "answer": answer,
            "image_id": info['image_path'],
        }
    