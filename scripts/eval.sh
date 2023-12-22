cd ..

dataset=semantic_kitti

phase=eval

tag=main
ckpt=epoch=99-step=95700
# miou:52.5416
# macc:60.0568

# tag=max
# ckpt=epoch=9-step=19130-v1
# miou:52.8217
# macc:60.0021

ckpt_path=/workspace/realtime/logs_semantic_kitti/minkprob_$tag/$ckpt.ckpt

CUDA_VISIBLE_DEVICES=1 \
    python $phase.py \
        --config=config/$dataset/$phase\_$tag.gin \
        --ckpt_path=$ckpt_path
echo minkprob_$tag/$ckpt.ckpt