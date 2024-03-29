import os
import torch
from dotenv import load_dotenv
from roboflow import Roboflow
from ultralytics import checks

def set_device():
    """
    Determines and sets the computation device based on availability.

    Checks for CUDA-compatible GPUs and uses them if available, otherwise falls back to CPU.
    Clears CUDA memory cache before setting the device to optimize memory usage.

    Returns:
        device (Union[str, int, list]): The device configuration for computation.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clears CUDA memory cache.
        device_count = torch.cuda.device_count()  # Get the number of available CUDA devices.
        if device_count > 1:
            # Use all available GPUs.
            device = list(range(device_count))  # Create a list of GPU IDs.
            device_names = ", ".join([torch.cuda.get_device_name(i) for i in device])  # Get names of all GPUs.
            print(f"Using GPUs: {device_names}")
        else:
            # Use the single available GPU.
            device = torch.cuda.current_device()  # Get the current GPU ID.
            print(f"Using GPU: {torch.cuda.get_device_name(device)}")  # Print the GPU name.
    else:
        # Fallback to CPU if no GPUs are available.
        device = 'cpu'
        print("Using CPU")
    return device

def download_dataset():
    """
    Downloads the specified dataset using Roboflow.

    Performs initial setup and checks, creates a dataset directory if it doesn't exist,
    and downloads the dataset for 'atta-leafcutter-ants-object-detection'.

    Returns:
        A tuple containing the current directory, project, and dataset information.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script.
    os.chdir(current_dir)  # Change the working directory to the script's directory.
    checks()  # Perform initial setup and checks.

    # Create a directory for datasets if not present and move into it.
    datasets_dir = os.path.join(current_dir, 'datasets')  # Define the path for the datasets directory.
    os.makedirs(datasets_dir, exist_ok = True)  # Create the datasets directory if it doesn't exist.
    os.chdir(datasets_dir)  # Change the working directory to the datasets directory.

    load_dotenv() # Load environment variables from .env file.
    
    # Initialize Roboflow, select the project and download the dataset.
    rf = Roboflow(api_key = os.getenv('ROBOFLOW_API_KEY'))  # Initialize the Roboflow instance with the API key.
    project = rf.workspace(os.getenv('ROBOFLOW_WORKSPACE')).project(os.getenv('ROBOFLOW_PROJECT'))  # Select the workspace and project.
    version = project.version(int(os.getenv('ROBOFLOW_VERSION')))
    dataset = version.download('yolov8')  # Download the specific version of the dataset for YOLOv8.

    os.chdir(current_dir) # Revert to the original working directory.

    return current_dir, project, dataset  # Return the current directory, project, and dataset information.