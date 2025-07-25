import requests
import base64

# 读取图片并转为 base64
with open('images/0/10868.png', 'rb') as f:
    img_base64 = base64.b64encode(f.read()).decode('utf-8')

data = {
    'image': img_base64,
    'text': 'doffehzyhshldl202507250216'
}

response = requests.post('http://10.119.18.90:5000/encode', json=data)
print(response.json())

# 可选：保存返回的图片
img_out_base64 = response.json().get('img_out_base64')
if img_out_base64:
    with open('encoded_out.png', 'wb') as f:
        f.write(base64.b64decode(img_out_base64))