import requests
import base64


# 读取图片并转为 base64
with open('images/0/10868.png', 'rb') as f:
    img_base64 = base64.b64encode(f.read()).decode('utf-8')

data = {
    'image': img_base64,
    # 'text': 'doffehzyhshldl202507250216'
    "text": "doffehzyhsh",
    'redundancy_rate': 3,
    'is_base64': True,
}

response = requests.post('http://localhost:30010/encode', json=data)
# print(response.json())

img_out_base64 = response.json().get('img_out_base64')
text_new = response.json().get('text_new')
print(text_new)
open('scripts/test_encode_out.png', 'wb').write(base64.b64decode(img_out_base64))
