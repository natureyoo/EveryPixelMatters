python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=$((RANDOM + 10000)) \
    tools/train_net_da.py \
    --config-file ./configs/da_ga_ca_cityscapes_VGG_16_FPN_4x.yaml \
    OUTPUT_DIR ./training_dir/GA_CondA_CLS_City2Foggy_VGG16