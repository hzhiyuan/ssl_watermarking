# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import os.path as osp
import base64
import uuid
import numpy as np
import torch
import random
from torchvision.transforms import ToPILImage

import data_augmentation
import encode
import evaluate
import utils
import utils_img

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_parser():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('Experiments parameters')
    aa("--data_dir", type=str, default="input/", help="Folder directory (Default: /input)")
    aa("--carrier_dir", type=str, default="carriers/", help="Directions of the latent space in which the watermark is embedded (Default: /carriers)")
    aa("--output_dir", type=str, default="output/", help="Output directory for logs and images (Default: /output)")
    aa("--save_images", type=utils.bool_inst, default=True, help="Whether to save watermarked images (Default: False)")
    aa("--evaluate", type=utils.bool_inst, default=True, help="Whether to evaluate the detector (Default: True)")
    aa("--decode_only", type=utils.bool_inst, default=False, help="To decode only watermarked images (Default: False)")
    aa("--verbose", type=int, default=1)

    group = parser.add_argument_group('Messages parameters')
    aa("--msg_type", type=str, default='bit', choices=['text', 'bit'], help="Type of message (Default: bit)")
    aa("--msg_path", type=str, default=None, help="Path to the messages text file (Default: None)")
    aa("--num_bits", type=int, default=None, help="Number of bits of the message. (Default: None)")
    aa("--dimentions", type=int, default=None, help="Number of dimensions of the latent space. (Default: None)")
    aa("--redundancy_rate", type=int, default=3, help="Redundancy rate of the binary message. (Default: 1)")
    aa("--is_base64", type=utils.bool_inst, default=True, help="Whether the message is base64 encoded. (Default: True)")
    
    group = parser.add_argument_group('Marking parameters')
    aa("--target_psnr", type=float, default=42.0, help="Target PSNR value in dB. (Default: 42 dB)")
    aa("--target_fpr", type=float, default=1e-6, help="Target FPR of the dectector. (Default: 1e-6)")

    group = parser.add_argument_group('Neural-Network parameters')
    aa("--model_name", type=str, default='resnet50', help="Marking network architecture. See https://pytorch.org/vision/stable/models.html and https://rwightman.github.io/pytorch-image-models/models/ (Default: resnet50)")
    aa("--model_path", type=str, default="models/dino_r50_plus.pth", help="Path to the model (Default: /models/dino_r50_plus.pth)")
    aa("--normlayer_path", type=str, default="normlayers/out2048_yfcc_orig.pth", help="Path to the normalization layer (Default: /normlayers/out2048.pth)")

    group = parser.add_argument_group('Optimization parameters')
    aa("--epochs", type=int, default=100, help="Number of epochs for image optimization. (Default: 100)")
    aa("--data_augmentation", type=str, default="all", choices=["none", "all"], help="Type of data augmentation to use at marking time. (Default: All)")
    aa("--optimizer", type=str, default="Adam,lr=0.01", help="Optimizer to use. (Default: Adam,lr=0.01)")
    aa("--scheduler", type=str, default=None, help="Scheduler to use. (Default: None)")
    aa("--batch_size", type=int, default=1, help="Batch size for marking. (Default: 128)")
    aa("--lambda_w", type=float, default=5e4, help="Weight of the watermark loss. (Default: 1.0)")
    aa("--lambda_i", type=float, default=1.0, help="Weight of the image loss. (Default: 1.0)")

    return parser


def init():
    parser = get_parser()
    params = parser.parse_args()

    # Set seeds for reproductibility
    torch.manual_seed(0)
    np.random.seed(0)

    params.model_path = 'models/dino_r50_plus.pth'
    params.normlayer_path = 'normlayers/out2048_yfcc_orig.pth'
    params.batch_size = 1
    params.target_psnr = 33
    params.msg_type = 'text'

    # Loads backbone and normalization layer
    if params.verbose > 0:
        print('>>> Building backbone and normalization layer...')
    backbone = utils.build_backbone(path=params.model_path, name=params.model_name)
    normlayer = utils.load_normalization_layer(path=params.normlayer_path)
    model = utils.NormLayerWrapper(backbone, normlayer)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    params.dimentions = model(torch.zeros((1,3,224,224)).to(device)).size(-1)

    return model, params

def load_carrier(params):
    # Load or generate carrier and angle
    if not os.path.exists(params.carrier_dir):
        os.makedirs(params.carrier_dir, exist_ok=True)
    carrier_path = os.path.join(params.carrier_dir,'carrier_%i_%i.pth'%(params.num_bits, params.dimentions))
    if os.path.exists(carrier_path):
        if params.verbose > 0:
            print('>>> Loading carrier from %s' % carrier_path)
        carrier = torch.load(carrier_path)
        assert params.dimentions == carrier.shape[1]
    else:
        if params.verbose > 0:
            print('>>> Generating carrier into %s... (can take up to a minute)' % carrier_path)
        carrier = utils.generate_carriers(params.num_bits, params.dimentions, output_fpath=carrier_path)
    carrier = carrier.to(device, non_blocking=True) # direction vectors of the hyperspace
    return carrier

def save_img(img_root, img_base64, params):
    req_id = str(uuid.uuid4())
    if img_root == 'data/original':
        img_dir = osp.join(img_root, req_id, '0')
    else:
        img_dir = osp.join(img_root, req_id)
    os.makedirs(img_dir, exist_ok=False)
    params.data_dir = osp.join(img_root, req_id)
    
    img_byte = base64.b64decode(img_base64.encode('utf-8'))
    open(osp.join(img_dir, f'image.jpg'), 'wb').write(img_byte)

    if img_root == 'data/original':
        params.output_dir = osp.join('data/encoded', req_id)
        os.makedirs(osp.join(params.output_dir, 'imgs'), exist_ok=False)
    else:
        params.output_dir = osp.join('data/decoded', req_id)

def decode_image(model, params, image, num_bits, redundancy_rate, is_base64):
    params.num_bits = num_bits
    params.redundancy_rate = redundancy_rate
    params.is_base64 = is_base64

    # 存图
    save_img('data/decoded', image, params)
    carrier = load_carrier(params)

    if params.verbose > 0:
        print('>>> Decoding watermarks...')

    df = evaluate.decode_multibit_from_folder(params.data_dir, carrier, model, params.msg_type, params.redundancy_rate, params.is_base64)
    df_path = os.path.join(params.output_dir,'decodings.csv')
    df.to_csv(df_path, index=False)
    if params.verbose > 0:
        print('Results saved in %s'%df_path)

    decoded_text = df.iloc[0]['msg']
    return decoded_text

def encode_image(model, params, image, text, redundancy_rate, is_base64):
    params.redundancy_rate = redundancy_rate
    params.is_base64 = is_base64
    params.num_bits = 8 * len(text) * params.redundancy_rate
    text_new = None
    if params.is_base64:
        if len(text) % 4 != 0:
            res = ''.join(random.choices([_ for _ in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'], k = 4 - len(text) % 4))
            text = text + res
            text_new = text
            params.num_bits = 8 * len(text) * params.redundancy_rate
        params.num_bits = params.num_bits // 4 * 3\
    
    # 存图
    save_img('data/original', image, params)
    carrier = load_carrier(params)

    # Load images
    if params.verbose > 0:
        print('>>> Loading images from %s...'%params.data_dir)
    dataloader = utils_img.get_dataloader(params.data_dir, batch_size=params.batch_size)

    # Generate messages
    if params.verbose > 0:
        print('>>> Loading messages...')
    # msgs = utils.load_messages(params.msg_path, params.msg_type, len(dataloader.dataset))
    msg = utils.string_to_binary(text, params.redundancy_rate, params.is_base64)
    msgs = [[int(i)==1 for i in msg]]

    msgs = torch.tensor(msgs)

    # Construct data augmentation
    data_aug = data_augmentation.All()

    # Marking
    if params.verbose > 0:
        print('>>> Marking images...')
    pt_imgs_out = encode.watermark_multibit(dataloader, msgs, carrier, model, data_aug, params)
    imgs_out = [ToPILImage()(utils_img.unnormalize_img(pt_img).squeeze(0)) for pt_img in pt_imgs_out] 
    
    imgs_dir = os.path.join(params.output_dir, 'imgs')
    if params.verbose > 0:
        print('>>> Saving images into %s...'%imgs_dir)
    imgs_out[0].save(os.path.join(imgs_dir, '0_out.png'))

    img_out_byte = open(osp.join(imgs_dir, '0_out.png'), 'rb').read()
    img_out_base64 = base64.b64encode(img_out_byte).decode('utf-8')
    return img_out_base64, text_new
