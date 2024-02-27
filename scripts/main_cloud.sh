cd ..

dataset=semantic_kitti

phase=main_cloud

# tag=main
# ckpt=epoch=99-step=95700

tag=max
ckpt=epoch=9-step=19130-v1

ckpt_path=/workspace/logs_semantic_kitti/minkprob_$tag/$ckpt.ckpt

CUDA_VISIBLE_DEVICES=0 \
    python $phase.py \
        --config=config/$dataset/inference\_$tag.gin \
        --ckpt_path=$ckpt_path