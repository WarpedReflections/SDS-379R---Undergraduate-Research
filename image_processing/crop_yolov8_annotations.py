import os
from PIL import Image
from roboflow import Roboflow

def normalize_bbox(width, height, bbox):
    """
    Converts bounding box coordinates from relative to absolute pixel values.

    Args:
        width (int): The width of the image.
        height (int): The height of the image.
        bbox (list): A list of bounding box coordinates in the format [x_center, y_center, bbox_width, bbox_height],
                     where each value is relative to the width and height.

    Returns:
        list: A list of bounding box coordinates in the format [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
              in absolute pixel values.
    """
    # Calculate absolute bounding box dimensions.
    x_center, y_center, bbox_width, bbox_height = (dim * size for dim, size in zip(bbox, (width, height, width, height)))
    top_left_x = x_center - (bbox_width / 2)
    top_left_y = y_center - (bbox_height / 2)
    return [top_left_x, top_left_y, top_left_x + bbox_width, top_left_y + bbox_height]

def crop_image(img, bbox, idx, cropped_path, image_file):
    """
    Crops an image according to the specified bounding box and saves it.

    Args:
        img (Image.Image): The image to crop.
        bbox (list): The bounding box with which to crop the image, specified as [top_left_x, top_left_y, bottom_right_x, bottom_right_y].
        idx (int): Index of the bounding box, used to generate unique file names.
        cropped_path (str): The directory path where the cropped images will be saved.
        image_file (str): The original image file name, used to generate the cropped image file name.

    This function does not return a value but saves the cropped image in the specified path.
    """
    cropped_img = img.crop(bbox)

    # Split the filename at the second underscore and use only the first two parts.
    base_name_parts = image_file.split('_', 2)[:2]
    base_name = '_'.join(base_name_parts)
    
    # Construct the cropped image name using the simplified base name.
    cropped_img_name = f"{base_name}_cropped_{idx}.jpg"
    cropped_img_path = os.path.join(cropped_path, cropped_img_name)
    cropped_img.save(cropped_img_path)

def process_images(base_dir):
    """
    Processes images in a specified base directory for cropping based on bounding box labels.

    For each 'test', 'train', and 'valid' folder, this function reads the label files, 
    calculates the absolute bounding box coordinates, and crops the images accordingly.

    Args:
        base_dir (str): The base directory path where the 'test', 'train', and 'valid' folders are located.

    This function does not return a value but results in cropped images being saved in a 'cropped_images' directory within each folder.
    """
    for folder in ['test', 'train', 'valid']:
        labels_path = os.path.join(base_dir, folder, 'labels')
        images_path = os.path.join(base_dir, folder, 'images')
        cropped_path = os.path.join(base_dir, folder, 'cropped_images')
        os.makedirs(cropped_path, exist_ok=True)

        for label_file in os.listdir(labels_path):
            image_file = label_file.replace('.txt', '.jpg') # Assuming image files are .jpg.
            image_path = os.path.join(images_path, image_file)
            if os.path.exists(image_path):
                with Image.open(image_path) as img, open(os.path.join(labels_path, label_file), 'r') as file:
                    bboxes = [list(map(float, line.split()))[1:] for line in file.readlines()] # Skip the class label.
                    for idx, bbox in enumerate(bboxes):
                        bbox_absolute = normalize_bbox(*img.size, bbox)
                        crop_image(img, bbox_absolute, idx, cropped_path, image_file)

def main():
    """
    Main function to orchestrate the downloading and processing of a dataset for image cropping based on bounding boxes.

    Sets the current working directory, downloads the dataset using Roboflow, and processes the images in the dataset.
    """
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
    dataset = project.version(10).download("yolov8")

    # Define the base directory where the dataset is located and process the images.
    yolov8_folder = next(os.walk(datasets_dir))[1][0] # Assumes there is only one directory inside 'datasets'.
    base_directory = os.path.join(datasets_dir, yolov8_folder)
    process_images(base_directory)

if __name__ == "__main__":
    main()  # Execute the main function if the script is run directly.