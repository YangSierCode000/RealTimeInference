# dataset=scannet
dataset=semantic_kitti

phase=eval
# config=res16unet34c

config=res16unet34c_prob
ckpt_path=/workspace/realtime/pretrained/cue.ckpt

# config=res16unet34c_probmg
# ckpt_path=/workspace/realtime/pretrained/cueplus.ckpt

CUDA_VISIBLE_DEVICES=1 \
    python $phase.py \
        --config=config/$dataset/$phase\_$config.gin \
        --ckpt_path=$ckpt_path