import torch

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