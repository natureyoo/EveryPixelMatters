python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=$((RANDOM + 10000)) \
    tools/train_net_da.py \
    --config-file ./configs/da_ga_ca_sim10k_VGG_16_FPN_4x.yaml \
    OUTPUT_DIR ./training_dir/GA_CondA_REG_TOP_SIM10k_VGG16_SMOOTH_ALPHA0.2_SOFT_PROB_TAU_0.1_WEAK_SUP
