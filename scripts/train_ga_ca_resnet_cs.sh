python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=$((RANDOM + 10000)) \
    tools/train_net_da.py \
    --config-file ./configs/da_ga_ca_cityscapes_R_101_FPN_4x.yaml \
    OUTPUT_DIR ./training_dir/RES_TMP