import os
import re
from PIL import Image
import torch
from minigpt4.common.config import Config
from minigpt4.common.eval_utils import prepare_texts, init_model, eval_parser
from minigpt4.conversation.conversation import CONV_VISION_minigptv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def parse_args():
    parser = eval_parser()
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--save_path", type=str, default="output", help="Path to save visualization")
    parser.add_argument("--res", type=int, default=100, help="Resolution of the image")
    return parser.parse_args()

def visualize_detection(image, boxes, labels, save_path):
    """
    Visualize detection results on the image
    """
    # Create figure and axes
    fig, ax = plt.subplots(1)
    
    # Display the image
    ax.imshow(image)
    
    # Create a color map for different classes
    colors = ['g' if label == 0 else 'r' for label in labels]
    
    # Add the bounding boxes and labels
    for box, color, label in zip(boxes, colors, labels):
        x, y, w, h = box
        rect = patches.Rectangle(
            (x, y), w-x, h-y,
            linewidth=2,
            edgecolor=color,
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label text
        label_text = "not-defect" if label == 0 else "defect"
        plt.text(x, y-10, label_text, color=color)
    
    # Save the visualization
    plt.axis('off')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def main():
    # Parse arguments
    args = parse_args()
    
    # Initialize model
    cfg = Config(args)
    model, vis_processor = init_model(args)
    model.eval()
    
    # Set up conversation template
    conv_temp = CONV_VISION_minigptv2.copy()
    conv_temp.system = ""
    
    # Load and process image
    image = Image.open(args.image_path).convert("RGB")
    width, height = image.size
    processed_image = vis_processor(image)
    
    # Prepare input
    question = "[detection] a defect or not-defect object and return the bounding boxes and its label. If not, bound around the object."
    texts = prepare_texts([question], conv_temp)
    
    # Generate prediction
    with torch.no_grad():
        answer = model.generate(
            processed_image.unsqueeze(0),
            texts,
            max_new_tokens=100,
            do_sample=False
        )[0]
    

    # Parse prediction
    answer = answer.replace("<unk>", "").replace(" ", "").strip()
    pred_boxes = []
    pred_labels = []
    
    pattern = r"<p>(.*?)<\/p>\{<(\d{1,3})><(\d{1,3})><(\d{1,3})><(\d{1,3})>\}"
    matches = re.finditer(pattern, answer)
    
    for match in matches:
        pred_class_str = match.group(1).strip()
        pred_class = 0 if "not-defect" in pred_class_str else 1
        
        bbox = [
            int(match.group(2)) / args.res * width,
            int(match.group(3)) / args.res * height,
            int(match.group(4)) / args.res * width,
            int(match.group(5)) / args.res * height
        ]
        
        pred_boxes.append(bbox)
        pred_labels.append(pred_class)
    
    # Visualize results
    save_path = os.path.join(args.save_path, "detection_result.png")
    visualize_detection(image, pred_boxes, pred_labels, save_path)
    
    # Print results
    print(f"Processed image: {args.image_path}")
    print(f"Found {len(pred_boxes)} objects")
    print(f"Visualization saved to: {save_path}")
    print("\nDetailed results:")
    for i, (box, label) in enumerate(zip(pred_boxes, pred_labels)):
        label_text = "not-defect" if label == 0 else "defect"
        print(f"Object {i+1}: {label_text} at coordinates {box}")

if __name__ == "__main__":
    main()


