import numpy as np
import cv2
import torch
import os
import time
import argparse
import pathlib

from model import create_model
from config import (
    NUM_CLASSES, DEVICE, CLASSES
)

np.random.seed(42)

# Construct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument(
    '-i', '--input', help='path to input video',
    default='data/number_plate/inference_data/video_1.mp4'
)
parser.add_argument(
    '--imgsz', 
    default=None,
    type=int,
    help='image resize shape'
)
parser.add_argument(
    '--threshold',
    default=0.25,
    type=float,
    help='detection threshold'
)
args = parser.parse_args()

#os.makedirs('inference_outputs/videos', exist_ok=True)

#add a directory to save frames
frame_output_dir='inference_outputs/frames'
os.makedirs(frame_output_dir, exist_ok=True)

COLORS = np.random.randint(0,255, size=(len(CLASSES),3)).tolist()

# Load the best model and trained weights.
model = create_model(
    num_classes=NUM_CLASSES
)
checkpoint = torch.load('outputs/best_model.pth', map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.to(DEVICE).eval()
print(model)

cap = cv2.VideoCapture(args.input)

if (cap.isOpened() == False):
    print('Error while trying to read video. Please check path again')

# Get the frame width and height.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

save_name = str(pathlib.Path(args.input)).split(os.path.sep)[-1].split('.')[0]
print(save_name)
# Define codec and create VideoWriter object .
out = cv2.VideoWriter(f"inference_outputs/videos/{save_name}.MOV", 
                      cv2.VideoWriter_fourcc(*'mp4v'), 30, 
                      (frame_width, frame_height))

frame_count = 0 # To count total frames.
total_fps = 0 # To get the final frames per second.

# Read until end of video.
while(cap.isOpened()):
    # Capture each frame of the video.
    ret, frame = cap.read()
    if ret:
        image = frame.copy()
        if args.imgsz is not None:
            image = cv2.resize(image, (args.imgsz, args.imgsz))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # Make the pixel range between 0 and 1.
        image /= 255.0
        # Bring color channels to front (H, W, C) => (C, H, W).
        image_input = np.transpose(image, (2, 0, 1)).astype(np.float32)
        # Convert to tensor.
        image_input = torch.tensor(image_input, dtype=torch.float).cuda()
        # Add batch dimension.
        image_input = torch.unsqueeze(image_input, 0)
        # Get the start time.
        start_time = time.time()
        # Predictions
        with torch.no_grad():
            outputs = model(image_input.to(DEVICE))
        end_time = time.time()
        
        # Get the current fps.
        fps = 1 / (end_time - start_time)
        # Total FPS till current frame.
        total_fps += fps
        frame_count += 1
        
        # Load all detection to CPU for further operations.
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
        # Carry further only if there are detected boxes.
        if len(outputs[0]['boxes']) != 0:
            boxes = outputs[0]['boxes'].data.numpy()
            scores = outputs[0]['scores'].data.numpy()
            # Filter out boxes according to `detection_threshold`.
            boxes = boxes[scores >= args.threshold].astype(np.int32)
            draw_boxes = boxes.copy()
            # Get all the predicited class names.
            pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
            
            # Draw the bounding boxes and write the class name on top of it.
            for j, box in enumerate(draw_boxes):
                class_name = pred_classes[j]
                try:
                    color = COLORS[CLASSES.index(class_name)]
                except ValueError:
                    print(f"Warning: Unknown class '{class_name}'-skipping drawing")
                    continue
                
                # Recale boxes.
                xmin = int((box[0] / image.shape[1]) * frame.shape[1])
                ymin = int((box[1] / image.shape[0]) * frame.shape[0])
                xmax = int((box[2] / image.shape[1]) * frame.shape[1])
                ymax = int((box[3] / image.shape[0]) * frame.shape[0])
                cv2.rectangle(frame,
                        (xmin, ymin),
                        (xmax, ymax),
                        color[::-1], 
                        3)
                cv2.putText(frame, 
                            class_name, 
                            (xmin, ymin-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.8, 
                            color[::-1], 
                            2, 
                            lineType=cv2.LINE_AA)
        # Calculate inference time in milliseconds
        inference_time = (end_time - start_time) * 1000
        cv2.putText(frame, f"{fps:.0f} FPS, {inference_time:.1f}ms", 
                    (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 
                    2, lineType=cv2.LINE_AA)
        #save current frame as image
        frame_filename= os.path.join(frame_output_dir, f"frame{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)

        cv2.imshow('image', frame)
        out.write(frame)
        # Press `q` to exit.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

# Release VideoCapture().
cap.release()
# Close all frames and video windows.
cv2.destroyAllWindows()

# Calculate and print the average FPS and inference time.
avg_fps = total_fps / frame_count
avg_inference_time = (1/avg_fps) * 1000  # Convert to milliseconds
print(f"Average FPS: {avg_fps:.3f}")
print(f"Average inference time: {avg_inference_time:.1f}ms")
