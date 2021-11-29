python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=$((RANDOM + 10000)) \
    tools/train_net.py \
    --config-file ./configs/sim10k_VGG_16_FPN_4x.yaml \
    OUTPUT_DIR ./training_dir/SourceOnly_SIM10k2CS_VGG16