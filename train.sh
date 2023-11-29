# dataset=scannet
dataset=semantic_kitti

phase=train
config=res16unet34c

python $phase.py --config=config/$dataset/$phase\_$config.gin
python $phase.py --config=config/$dataset/$phase\_$config.gin