import gradio as gr
from functions import *

title = "Water Body Segmentation - Image Segmentation PyTorch"
examples = ['examples/image1.png', 'examples/image2.png', 'examples/image3.png', 'examples/image4.png', 'examples/image5.png']

interface = gr.Interface(fn=predict, inputs=gr.Image(type= 'numpy').style(height= 256),
            outputs= gr.Image(type = "numpy").style(height= 256),
            examples= examples, title= title, css= '.gr-box {background-color: rgb(230 230 230);}')

interface.launch()