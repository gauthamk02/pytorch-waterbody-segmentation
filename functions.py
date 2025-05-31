import cv2
import numpy as np

def resize_preserve_aspect_ratio(image, size):
    h, w = image.shape[:2]
    if h > w:
        image = cv2.resize(image, (size * w // h, size))
    else:
        image = cv2.resize(image, (size, size * h // w))
    return image

def run_inference(inp_image, session, input_name, output_name):
    inp_dim = inp_image.shape[:2]

    image = cv2.resize(inp_image, (256, 256))
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)

    pred_onx = session.run([output_name], {input_name: image.astype(np.float32)})[0]
    pred_onx = pred_onx > 0.5
    pred_onx = pred_onx * 255

    pred_onx = cv2.resize(pred_onx[0, 0].astype(np.uint8) , (inp_dim[1], inp_dim[0]))
    pred_onx = np.expand_dims(pred_onx, axis=2)
    pred_onx = np.concatenate((pred_onx, pred_onx, pred_onx), axis=2)

    output = resize_preserve_aspect_ratio(pred_onx, 400)
    return output