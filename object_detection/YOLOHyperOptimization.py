import os
from device_manager import set_device 
from ultralytics import YOLO

def tune_model_on_custom_dataset(dataset_path, device):
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
    model = YOLO('yolov8s.pt')  # Initialize the YOLO model specifying the device.
    
    # Start tuning process for hyperparameters on the custom dataset.
    model.tune(data = dataset_path,
               epochs = 100,
               iterations = 10,
               optimizer = 'AdamW',
               plots = True,
               save = True,
               val = True,
               device = device)

def main():
    """
    Main function to set up the environment and tune a YOLO model on a custom dataset.

    This function sets the current working directory to the script's location to ensure that all relative 
    paths are correctly interpreted. It then defines the path to the custom dataset configuration and determines 
    the best computation device available. Finally, it calls the tune_model_on_custom_dataset function to 
    tune a YOLO model on this dataset using the determined device.

    The main function doesn't accept any arguments and doesn't return any value. It serves to orchestrate 
    the model tuning process.
    """
    # Set the current working directory to the script's location to ensure relative paths are handled correctly.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)

    # Define the path to the dataset configuration file.
    dataset_path = os.path.join(current_dir, "datasets/Atta-Leafcutter-Ants-Object-Detection-9/data.yaml")

    # Determine the best device to use based on GPU availability and system configuration.
    device = set_device()

    # Tune the model on the custom dataset using the determined device.
    tune_model_on_custom_dataset(dataset_path, device)

if __name__ == "__main__":
    main()  # Execute the main function if the script is run directly.