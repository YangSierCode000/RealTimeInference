# dataset=scannet
dataset=semantic_kitti

phase=train

# config=res16unet34c
config=res16unet34c_prob
tag=1202_220411
id=ke29omho

ckpt=/workspace/realtime/logs_semantic_kitti/minkprob_$tag/last.ckpt

# config=res16unet34c_probmg

CUDA_VISIBLE_DEVICES=0 \
    python $phase.py \
        --config=config/$dataset/$phase\_$config.gin \
        --ckpt=$ckpt \
        --tag=$tag \
        --id=$id
