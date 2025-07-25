# bits
# python main_multibit.py \
#   --data_dir images \
#   --model_path models/dino_r50_plus.pth \
#   --normlayer_path normlayers/out2048_coco_orig.pth \
#   --batch_size 4 \
#   --target_psnr 33 \
#   --num_bits 50

# text
python main_multibit.py \
  --data_dir images \
  --model_path models/dino_r50_plus.pth \
  --normlayer_path normlayers/out2048_coco_orig.pth \
  --batch_size 4 \
  --target_psnr 33 \
  --msg_type text \
  --msg_path messages/msgs.txt