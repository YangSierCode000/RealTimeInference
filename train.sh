dataset=semantic_kitti

phase=train
tag=main
id=0.4-$tag

ckpt=epoch=99-step=95700
ckpt=/workspace/realtime/logs_semantic_kitti/minkprob_$tag/$ckpt.ckpt

CUDA_VISIBLE_DEVICES=0 \
    python $phase.py \
        --config=config/$dataset/$phase\_$tag.gin \
        --ckpt=$ckpt \
        --tag=$tag \
        --id=$id \
