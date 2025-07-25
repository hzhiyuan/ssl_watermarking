import requests
import base64

# 读取带水印的图片并转为 base64
# with open('data/encoded/6a539477-12cf-4b3b-855a-41137ae2df69/imgs/0_out.png', 'rb') as f:
with open('scripts/test_encode_out.png', 'rb') as f:
    img_base64 = base64.b64encode(f.read()).decode('utf-8')

secret = 'doffehzyhsh8'
redundancy_rate = 3
is_base64 = True
num_bits = 8 * len(secret) * redundancy_rate
if is_base64:
    num_bits = num_bits // 4 * 3

data = {
    'image': img_base64,
    'num_bits': num_bits,
    'redundancy_rate': redundancy_rate,
    'is_base64': is_base64,
}

response = requests.post('http://localhost:30010/decode', json=data)
print(response.json())