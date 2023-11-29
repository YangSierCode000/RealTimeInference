# dataset=scannet
dataset=semantic_kitti

phase=eval
config=res16unet34c

ckpt_path=/home/haobo/workspace/realtime/pretrained/cue.ckpt

python $phase.py --config=config/$dataset/$phase\_$config.gin --ckpt_path=$ckpt_path