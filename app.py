import gradio as gr
from functions import *

title = "Water Body Segmentation - Image Segmentation PyTorch"
examples = ['samples/image1.png', 'samples/image2.png', 'samples/image3.png', 'samples/image4.png', 'samples/image5.png']

interface = gr.Interface(fn=predict, inputs=gr.Image(type= 'numpy').style(height= 256),
            outputs= gr.Image(type = "numpy").style(height= 256),
            examples= examples, title= title, css= '.gr-box {background-color: rgb(230 230 230);}')

interface.launch()