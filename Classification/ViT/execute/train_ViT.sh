setsid python ViT.py \
  --data dataset \
  --arch ViT-B_16 \
  --pretrained \
  --epochs 50 \
  --batch-size 8 \
  --lr 0.0003 \
  --image-size 224 \
  --num-classes 2 \
  --logdir info/logs \
  --save-path info/models/vit_glas_best.pth \
  > info/debugs/Vit.log 2>&1 &