setsid python3 ResNet.py \
  --data dataset \
  --arch ResNet-50 \
  --pretrained \
  --epochs 50 \
  --batch-size 8 \
  --lr 0.0003 \
  --image-size 224 \
  --num-classes 2 \
  --logdir info/logs \
  --save-path info/models/ResNet_glas_best.pth \
##  > info/debugs/ResNet.log 2>&1 &