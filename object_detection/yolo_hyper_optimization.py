import os
from gpu_roboflow_setup import set_device, download_dataset
from ultralytics import YOLO

def tune_model_on_custom_dataset(current_dir, dataset_path, device):
    """
    Tunes a YOLO model on a custom dataset to find optimal hyperparameters.

    Initializes the YOLOv8 model and performs hyperparameter tuning on the specified dataset. 
    The tuning process involves adjusting various parameters to improve model performance based on 
    validation set accuracy. The results, including plots of the tuning process, are saved for analysis.

    Args:
        dataset_path (str): The path to the dataset configuration file, typically a YAML file that 
            specifies training and validation data locations and classes.
        device (str or torch.device): The device on which to perform the tuning. Can be 'cpu' or 
            a specific GPU ('cuda:0', 'cuda:1', etc.) as determined by the set_device function.

    The function doesn't return any value explicitly but results in saved files that detail the tuning process 
    and potentially updates the model weights to those tuned for the given dataset.
    """
    model = YOLO('yolov8s.pt')  # Initialize the base model. (Used for the first run.)
    model = YOLO(f'{current_dir}/runs/first_run/custom_train/weights/best.pt') # Initialize the custom model from the first run.
    
    # Start tuning process for hyperparameters on the custom dataset.
    model.tune(data = dataset_path, 
               epochs = 30, 
               iterations = 300, 
               optimizer = 'AdamW', 
               plots = True, 
               save = True, 
               val = True, 
               device = device)

def main():
    """
    Orchestrates the tuning of a YOLO model on a custom dataset by setting up the environment and managing the workflow.
    """
    # Determine the best device to use based on GPU availability and system configuration.
    device = set_device()

    # Acquire the dataset and its path.
    current_dir, _, dataset = download_dataset()

    # Specify the dataset configuration file path.
    dataset_path = os.path.join(dataset.location, 'data.yaml')

    # Tune the model on the custom dataset using the determined device.
    tune_model_on_custom_dataset(current_dir, dataset_path, device)

if __name__ == "__main__":
    main()  # Execute the main function if the script is run directly.