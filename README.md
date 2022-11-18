---
title: Water Body Segmentation
emoji: 🤗
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 3.10.0
app_file: app.py
pinned: false
---

# UNET Water Body Segmentation - PyTorch

This project contains the code for training and deploying a UNET model for water body segmentation from satellite images. The model is trained on the [Satellite Images of Water Bodies](https://www.kaggle.com/datasets/franciscoescobar/satellite-images-of-water-bodies) from Kaggle. The model is trained using PyTorch and deployed using [Gradio](https://gradio.app/) on [Hugging Face Spaces](https://huggingface.co/spaces).

## 🚀 Getting Started

All the code for training the model and exporting to ONNX format is present in the [notebook](notebooks) folder or you can use this [Kaggle Notebook](https://www.kaggle.com/code/gauthamkrishnan119/water-body-segmentation-pytorch) for training the model. The [app.py](app.py) file contains the code for deploying the model using Gradio. 

## 🤗 Demo

You can try out the model on [Hugging Face Spaces](https://huggingface.co/spaces/gauthamk/water-body-segmentation)

## 🖥️ Sample Inference

![Sample Inference](samples/sample1.png)

