# Import necessary libraries
from ultralytics import checks
from roboflow import Roboflow
import os
import random
import subprocess
import torch

torch.cuda.set_device(0)

# Set the device to GPU if available, otherwise fallback to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Perform checks using the ultralytics library.
checks()

# Set the current working directory to the script's location.
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

# Create a directory for datasets if it doesn't exist, and set the current working directory to it.
datasets_dir = os.path.join(current_dir, 'datasets')
os.makedirs(datasets_dir, exist_ok=True)
os.chdir(datasets_dir)

# Initialize Roboflow and download the dataset.
rf = Roboflow(api_key="VWSCEQrXBqZ0RQBWkwCS")
project = rf.workspace("sds-379r").project("atta-leafcutter-ants-object-detection")
dataset = project.version(9).download("yolov8")

# Custom Training with YOLOv8.
os.chdir(current_dir) # Set the current working directory to the script's location.
# Train the YOLOv8 model on the custom dataset.
subprocess.run(["yolo", "task=detect", "mode=train", f"model=yolov8s.pt", f"data={dataset.location}/data.yaml", "epochs=100", 
                "imgsz=800", "plots=True", f"device={device}"], check=True)

# List the contents in the training directory to verify.
print(os.listdir(os.path.join(current_dir, 'runs/detect/train/')))

# Validate the custom-trained model.
subprocess.run(["yolo", "task=detect", "mode=val", f"model={current_dir}/runs/detect/train/weights/best.pt", 
                f"data={dataset.location}/data.yaml"], check=True)

# Inference with the custom-trained model.
subprocess.run(["yolo", "task=detect", "mode=predict", f"model={current_dir}/runs/detect/train/weights/best.pt", "conf=0.25", 
                f"source={dataset.location}/test/images", "save=True"], check=True)

# Deploy the trained model on Roboflow.
project.version(dataset.version).deploy(model_type="yolov8", model_path=os.path.join(current_dir, "runs/detect/train/"))

# Load the deployed model for inference.
model = project.version(dataset.version).model

# Perform inference on a random image from the test set.
test_set_loc = os.path.join(dataset.location, "test/images/")
random_test_image = random.choice(os.listdir(test_set_loc))
print("Running inference on " + random_test_image)
pred = model.predict(os.path.join(test_set_loc, random_test_image), confidence=40, overlap=30).json()
pred