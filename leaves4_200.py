import argparse

import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', default='ae.jpg', help='image')
    parser.add_argument('-m', '--model', default='model.tflite', help='.tflite model')
    args = parser.parse_args()

    interpreter = tflite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    floating_model = input_details[0]['dtype'] == np.float32  # just check input tensor dtype
    
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
      
    img = Image.open(args.image).resize((width, height))
    input_data = np.expand_dims(img, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])    
    results = np.squeeze(output_data)
    ans = results.argsort()[-5:][::-1][0]
    print(['青楓', '水同木', '大葉山欖', '盾柱木'][ans])
