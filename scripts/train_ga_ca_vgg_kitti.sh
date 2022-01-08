python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=$((RANDOM + 10000)) \
    tools/train_net_da.py \
    --config-file ./configs/da_ga_ca_kitti_VGG_16_FPN_4x.yaml \
    OUTPUT_DIR ./training_dir/GA_CondA_REG_TOP_KITTI_VGG16_DIFFERENT_THRESHOLD_SMOOTH_ALPHA0.2_SOFT_PROB_TAU_0.1