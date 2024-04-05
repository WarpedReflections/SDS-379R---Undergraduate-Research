import os
from gpu_roboflow_setup import set_device, download_dataset
from ultralytics import YOLO

def model_train(base_model, train_validation_parameters, current_dir):
    """
    Trains a YOLOv8 model on the specified dataset.

    Args:
        base_model (YOLO): The YOLOv8 model to be trained.
        train_validation_parameters (dict): A dictionary containing training parameters.
        current_dir (str): The directory to store the training outputs.

    Trains the model using custom parameters and saves the trained model in a specified directory.
    """
    # Train the model with specified parameters and directory structure for saving the model.(Used for the first run).
    # base_model.train(train_validation_parameters, epochs = 300, project = f'{current_dir}/runs/detect', name = "custom_train", 
    #                  optimizer = 'AdamW', lr0 = 0.00769, lrf = 0.01161, momentum = 0.98, weight_decay = 0.00049, warmup_epochs = 3.0976, 
    #                  warmup_momentum = 0.81893, box = 7.5, cls = 0.72963, dfl = 1.87382, hsv_h = 0.0125, hsv_s = 0.70358, 
    #                  hsv_v = 0.39202, degrees = 0.0, translate = 0.07629, scale = 0.3911, shear = 0.0, perspective = 0.0, flipud = 0.0, 
    #                  fliplr = 0.535, mosaic = 0.84149, mixup = 0.0, copy_paste = 0.0)
    
    # Train the model with specified parameters and directory structure for saving the model.
    base_model.train(**train_validation_parameters, epochs = 300, project = f'{current_dir}/runs/detect', name = "custom_train", 
                     optimizer = 'AdamW', lr0 = 0.0107, lrf = 0.00719, momentum = 0.9209, weight_decay = 0.00062, warmup_epochs = 0.91169, 
                     warmup_momentum = 0.88237, box = 7.60778, cls = 0.37221, dfl = 1.61228, hsv_h = 0.0149, hsv_s = 0.58918, 
                     hsv_v = 0.29909, degrees = 0.0, translate = 0.10301, scale = 0.46502, shear = 0.0, perspective = 0.0, flipud = 0.0, 
                     fliplr = 0.42842, mosaic = 0.55417, mixup = 0.0, copy_paste = 0.0)

def model_validation(custom_model, train_validation_parameters):
    """
    Validates the custom-trained YOLOv8 model.

    Args:
        custom_model (YOLO): The custom-trained model to be used for validation.
        train_validation_parameters (dict): A dictionary containing validation parameters.

    Performs validation using the specified model and parameters, and saves the results.
    """
    custom_model.val(**train_validation_parameters, save_json = True, save_hybrid = True)  # Validate the model and save results.

def model_prediction(custom_model, core_parameter, dataset):
    """
    Performs inference using the custom-trained model on a test dataset.

    Args:
        custom_model (YOLO)): The custom-trained model to be used for prediction.
        core_parameter (dict): A dictionary containing the computation device.
        dataset: The dataset on which the model is to perform inference.

    Saves predictions, confidence scores, and optionally cropped images.
    """
    # Predict using the custom model on the specified dataset and save the results.
    custom_model.predict(**core_parameter, source = f'{dataset.location}/test/images', save = True, save_txt = True, save_conf = True,
                         save_crop = True)
    
def model_deployment(project, dataset, current_dir):
    """
    Deploys the trained YOLOv8 model on Roboflow.

    Args:
        project: The Roboflow project to which the model belongs.
        dataset: The dataset used for training the model.
        current_dir (str): The directory containing the trained model.

    Deploys the model for online or edge deployment.
    """
    # Deploy the trained model for use in Roboflow applications.
    project.version(dataset.version).deploy(model_type = 'yolov8', model_path = os.path.join(current_dir, 'runs/detect/custom_train/'))

def main():
    """
    Main function to execute the model training, validation, prediction, and deployment sequence.

    Sets up the device, downloads the dataset, initializes the model, and performs
    training, validation, prediction, and deployment using the specified parameters.
    """
    device = set_device()  # Set the computation device.
    current_dir, project, dataset = download_dataset()  # Download the dataset.

    core_parameter = {
        'device': device,  # Specify the computation device for model operations.
    }

    train_validation_parameters = {
        'data': f'{dataset.location}/data.yaml',  # Specify the dataset location.
        'plots': True,  # Enable plotting of training and validation metrics.
        **core_parameter,  # Include core parameters in training and validation parameters.
    }
    
    # base_model = YOLO('yolov8s.pt')  # Initialize the base model. (Used for the first run.)
    # base_model = YOLO(f'{current_dir}/runs/first_run/custom_train/weights/best.pt')  # Initialize the model from the first run.
    # model_train(base_model, train_validation_parameters, current_dir)  # Train the model.

    custom_model = YOLO(f'{current_dir}/runs/detect/custom_train/weights/best.pt') # Load the model from the best weights.
    model_validation(custom_model, train_validation_parameters)  # Validate the model.
    model_prediction(custom_model, core_parameter, dataset)  # Run predictions with the model.
    model_deployment(project, dataset, current_dir)  # Deploy the model.

if __name__ == "__main__":
    main()  # Execute the main function if the script is run directly.