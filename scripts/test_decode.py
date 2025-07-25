import requests
import base64

# 读取带水印的图片并转为 base64
with open('data/encoded/6a539477-12cf-4b3b-855a-41137ae2df69/imgs/0_out.png', 'rb') as f:
# with open('images/0/10868_screenshot_wx.png', 'rb') as f:
    img_base64 = base64.b64encode(f.read()).decode('utf-8')

data = {
    'image': img_base64,
    'num_bits': 208
}

response = requests.post('http://localhost:30010/decode', json=data)
print(response.json())