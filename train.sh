# dataset=scannet
dataset=semantic_kitti

phase=train

# config=res16unet34c
config=res16unet34c_prob
# config=res16unet34c_probmg

CUDA_VISIBLE_DEVICES=1 \
    python $phase.py \
        --config=config/$dataset/$phase\_$config.gin \
