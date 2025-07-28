import os
import base64
from flask import Flask, request, jsonify
import interface

app = Flask(__name__)

# 全局变量
params = None
model = None


def init_global():
    global params, model
    model, params = interface.init()

class ReqObj:
    def __init__(self, image, text=""):
        self.image = image
        self.text = text

@app.route('/encode', methods=['POST'])
def encode_api():
    data = request.get_json()
    img_base64 = data['image']
    text = data['text']
    redundancy_rate = data.get('redundancy_rate', 3)
    is_base64 = data.get('is_base64', False)

    img_out_base64, text_new = interface.encode_image(model, params, img_base64, text, redundancy_rate, is_base64)
    return jsonify({'img_out_base64': img_out_base64, 'text_new': text_new})

@app.route('/decode', methods=['POST'])
def decode_api():
    data = request.get_json()
    img_base64 = data['image']
    num_bits = data['num_bits']
    redundancy_rate = data.get('redundancy_rate', 3)
    is_base64 = data.get('is_base64', False)

    decoded_text = interface.decode_image(model, params, img_base64, num_bits, redundancy_rate, is_base64)
    print(decoded_text)
    return jsonify({'decoded_text': decoded_text})

if __name__ == '__main__':
    init_global()
    app.run(host='0.0.0.0', port=30010)