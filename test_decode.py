import requests
import base64

# 读取带水印的图片并转为 base64
with open('encoded_out.png', 'rb') as f:
    img_base64 = base64.b64encode(f.read()).decode('utf-8')

data = {
    'image': img_base64,
    'num_bits': 208
}

response = requests.post('http://10.119.18.90:5000/decode', json=data)
print(response.json())