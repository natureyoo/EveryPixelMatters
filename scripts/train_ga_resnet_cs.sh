python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=$((RANDOM + 10000)) \
    tools/train_net_da.py \
    --config-file ./configs/da_ga_cityscapes_R_101_FPN_4x.yaml \
    OUTPUT_DIR ./training_dir/GA_City2Foggy_R101_lr0.02