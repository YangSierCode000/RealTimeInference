cd ..

dataset=semantic_kitti

phase=inference

# tag=main
# ckpt=epoch=99-step=95700

# tag=max
# ckpt=epoch=99-step=95700

tag=mini
ckpt=epoch=74-step=28725

ckpt_path=/workspace/logs_semantic_kitti/minkprob_$tag/$ckpt.ckpt

CUDA_VISIBLE_DEVICES=1 \
    python $phase.py \
        --config=config/$dataset/$phase\_$tag.gin \
        --ckpt_path=$ckpt_path