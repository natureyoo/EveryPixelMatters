python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=$((RANDOM + 10000)) \
    tools/train_net_da.py \
    --config-file ./configs/da_ga_ca_cityscapes_VGG_16_FPN_4x.yaml \
    SOLVER.MAX_ITER 20000 \
    SOLVER.IMS_PER_BATCH 16 \
    MODEL.ADV.COND_CLASS True \
    MODEL.WEIGHT ./da_ga_cs_vgg16_final.pth \
    OUTPUT_DIR ./training_dir/GA_CondA_CLS_City2Foggy_VGG16