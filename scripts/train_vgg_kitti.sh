python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=$((RANDOM + 10000)) \
    tools/train_net.py \
    --config-file ./configs/kitti_VGG_16_FPN_4x.yaml \
    OUTPUT_DIR ./training_dir/SourceOnly_Kitti2CS_VGG16_no_gradientclip