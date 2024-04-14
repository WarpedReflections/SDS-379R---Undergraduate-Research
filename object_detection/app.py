import PIL.Image as Image
import gradio as gr
from ultralytics import YOLO

model = YOLO('./runs/detect/custom_train/weights/best.pt') # Load the fine-tuned Atta leafcutter ant model.

def predict_image(img, conf_threshold, iou_threshold):
    """
    Predicts Atta leafcutter ants in an image using the fine-tuned Atta leafcutter ant model.

    Args:
        img (PIL.Image): The input image for prediction.
        conf_threshold (float): The confidence threshold for detection.
        iou_threshold (float): The Intersection over Union threshold for detection.

    Returns:
        PIL.Image: The input image annotated with bounding boxes and labels of detected objects.

    Performs object detection on the provided image using specified confidence and IoU thresholds.
    The results are visualized on the image itself.
    """
    results = model.predict(
        source = img,
        conf = conf_threshold,
        iou = iou_threshold,
        show_labels = True,
        show_conf = True,
        imgsz = 640,
    )

    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])

    return im

iface = gr.Interface(
    fn = predict_image,
    inputs = [
        gr.Image(type = 'pil', label = "Upload Image"),
        gr.Slider(minimum = 0, maximum = 1, value = 0.25, label = "Confidence Threshold"),
        gr.Slider(minimum = 0, maximum = 1, value = 0.45, label = "IoU Threshold")
    ],
    outputs = gr.Image(type = 'pil', label = 'Result'),
    title = "Ultralytics YOLOv8 Object Detection Gradio App",
    description = "Upload images for inference. A fine-tuned Atta leafcutter ant model is used by default.",
    examples = [
        ['./datasets/Atta-Leafcutter-Ants-Object-Detection-11/test/images/iNAT1605043_1_jpg.rf.5bfef509f933b211f38f07255011150a.jpg', 
         0.25, 0.45],
        ['./datasets/Atta-Leafcutter-Ants-Object-Detection-11/test/images/iNAT1795742_1_jpg.rf.1d316baf1f05a310f10aa39e73d51805.jpg', 
         0.25, 0.45],
    ]
)

if __name__ == '__main__':
    iface.launch()  # Launch the Gradio interface for the YOLOv8 object detection app.