import os
import cv2
import time
import torch
import glob
import numpy as np
import xml.etree.ElementTree as ET
import argparse
import csv
from torchvision import transforms
from model import create_model
from config import NUM_CLASSES, DEVICE, CLASSES

# ========== Helper Classes and Functions ==========
class Rectangle:
    def __init__(self, x1, y1, x2, y2):
        self.x1, self.y1 = min(x1, x2), min(y1, y2)
        self.x2, self.y2 = max(x1, x2), max(y1, y2)

    def area(self):
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    def intersection(self, other):
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        if x1 >= x2 or y1 >= y2:
            return 0.0
        return (x2 - x1) * (y2 - y1)

    def iou(self, other):
        inter = self.intersection(other)
        union = self.area() + other.area() - inter
        return inter / union if union > 0 else 0.0

def infer_transforms(image):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])(image)

def write_voc_xml(filename, image_shape, boxes, scores, labels, save_dir):
    h, w, _ = image_shape
    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'filename').text = filename
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(w)
    ET.SubElement(size, 'height').text = str(h)
    ET.SubElement(size, 'depth').text = '3'

    for box, score, label in zip(boxes, scores, labels):
        obj = ET.SubElement(annotation, 'object')
        ET.SubElement(obj, 'name').text = label
        ET.SubElement(obj, 'score').text = str(score)
        bndbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(box[0])
        ET.SubElement(bndbox, 'ymin').text = str(box[1])
        ET.SubElement(bndbox, 'xmax').text = str(box[2])
        ET.SubElement(bndbox, 'ymax').text = str(box[3])

    tree = ET.ElementTree(annotation)
    os.makedirs(save_dir, exist_ok=True)
    tree.write(os.path.join(save_dir, f"{filename}.xml"))

def parse_voc_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        box = Rectangle(
            int(bbox.find("xmin").text),
            int(bbox.find("ymin").text),
            int(bbox.find("xmax").text),
            int(bbox.find("ymax").text)
        )
        box.label = obj.find("name").text.lower()
        box.score = float(obj.find("score").text) if obj.find("score") is not None else 1.0
        boxes.append(box)
    return boxes

def evaluate(pred_dir, gt_dir):
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith(".xml")])
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(".xml")])
    all_files = sorted(set(os.path.splitext(f)[0] for f in pred_files + gt_files))

    total_tp = total_fp = total_fn = 0
    total_iou_sum = 0.0
    total_iou_count = 0

    for name in all_files:
        pred_path = os.path.join(pred_dir, name + ".xml")
        gt_path = os.path.join(gt_dir, name + ".xml")

        pred_boxes = parse_voc_xml(pred_path) if os.path.exists(pred_path) else []
        gt_boxes = parse_voc_xml(gt_path) if os.path.exists(gt_path) else []

        matched_gt = set()
        tp = fp = 0
        iou_sum = iou_count = 0

        for pred in pred_boxes:
            matched = False
            for i, gt in enumerate(gt_boxes):
                if i in matched_gt: continue
                if pred.label == gt.label and pred.iou(gt) > 0.0:
                    tp += 1
                    matched_gt.add(i)
                    iou_sum += pred.iou(gt)
                    iou_count += 1
                    matched = True
                    break
            if not matched:
                fp += 1

        fn = len(gt_boxes) - len(matched_gt)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_iou_sum += iou_sum
        total_iou_count += iou_count

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    avg_iou = total_iou_sum / total_iou_count if total_iou_count else 0.0

    print("\n========= Evaluation Results =========")
    print(f"TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
    print(f"Precision (mAP@0.5): {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Average IoU: {avg_iou:.4f}")

# ========== Main Function ==========
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='outputs/best_model.pth')
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('--imgsz', default=None, type=int)
    parser.add_argument('--threshold', default=0.25, type=float)
    parser.add_argument('--gt_dir', required=True)
    parser.add_argument('--pred_xml_dir', default='inference_outputs/xmls')
    args = parser.parse_args()

    model = create_model(num_classes=NUM_CLASSES)
    checkpoint = torch.load(args.weights, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE).eval()

    os.makedirs(args.pred_xml_dir, exist_ok=True)
    test_images = [img for ext in ['jpg', 'jpeg', 'png'] for img in glob.glob(f"{args.input}/*.{ext}")]

    for img_path in test_images:
        name = os.path.splitext(os.path.basename(img_path))[0]
        image = cv2.imread(img_path)
        orig = image.copy()
        if args.imgsz: image = cv2.resize(image, (args.imgsz, args.imgsz))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = torch.unsqueeze(infer_transforms(image), 0).to(DEVICE)

        with torch.no_grad():
            outputs = model(image_tensor)
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

        if len(outputs[0]['boxes']) > 0:
            boxes = outputs[0]['boxes'].data.numpy()
            scores = outputs[0]['scores'].data.numpy()
            labels = [CLASSES[i] for i in outputs[0]['labels'].numpy()]
            keep = scores >= args.threshold
            write_voc_xml(
                filename=name,
                image_shape=orig.shape,
                boxes=boxes[keep].astype(int),
                scores=scores[keep],
                labels=np.array(labels)[keep],
                save_dir=args.pred_xml_dir
            )
        print(f"Processed {name}")

    evaluate(pred_dir=args.pred_xml_dir, gt_dir=args.gt_dir)
'''python merged_script.py \
  --weights outputs/best_model.pth \
  -i /path/to/input/images \
  --gt_dir /path/to/ground_truth/xmls \
  --threshold 0.5
'''
