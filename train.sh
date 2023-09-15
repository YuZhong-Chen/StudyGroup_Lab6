python main.py \
    --model deeplabv3plus_mobilenet \
    --gpu_id 0 \
    --crop_val \
    --crop_size 513 \
    --lr 0.01 \
    --batch_size 8 \
    --val_interval 5 \
    --output_stride 16