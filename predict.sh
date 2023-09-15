# python predict.py \
#     --input test_inputs/test1.png \
#     --dataset cityscapes \
#     --model deeplabv3plus_mobilenet \
#     --ckpt weights/best_deeplabv3plus_mobilenet_cityscapes_os16.pth \
#     --save_val_results_to test_results

python predict.py \
    --input test_inputs/test1.png \
    --dataset cityscapes \
    --model deeplabv3plus_resnet101 \
    --ckpt weights/best_deeplabv3plus_resnet101_cityscapes_os16.pth \
    --save_val_results_to test_results