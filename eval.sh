dataset=semantic_kitti

phase=eval
tag=main
id=0.3-$tag

ckpt=epoch=99-step=95700
ckpt_path=/workspace/realtime/logs_semantic_kitti/minkprob_$tag/$ckpt.ckpt

CUDA_VISIBLE_DEVICES=0 \
    python $phase.py \
        --config=config/$dataset/$phase\_$tag.gin \
        --ckpt_path=$ckpt_path