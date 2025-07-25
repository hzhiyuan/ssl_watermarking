import requests
import base64

# 读取图片并转为 base64
with open('images/0/10868.png', 'rb') as f:
    img_base64 = base64.b64encode(f.read()).decode('utf-8')

data = {
    'image': img_base64,
    'text': 'doffehzyhshldl202507250216'
}

response = requests.post('http://localhost:30010/encode', json=data)
# print(response.json())

# img_out_base64 = response.json().get('img_out_base64')
