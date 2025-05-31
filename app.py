import gradio as gr
import onnxruntime as rt
from functions import run_inference

model_path = 'weights/model.onnx'
session = rt.InferenceSession(model_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

title = "Water Body Segmentation - Image Segmentation PyTorch"
examples = ['examples/image1.png', 'examples/image2.png', 'examples/image3.png', 'examples/image4.png', 'examples/image5.png']

def inference_wrapper(image):
    return run_inference(image, session, input_name, output_name)


interface = gr.Interface(fn=inference_wrapper, 
            inputs=gr.Image(type='numpy', height=400, width=400),
            outputs=gr.Image(type="numpy", height=400, width=400),
            examples=examples, 
            title=title)

interface.launch()