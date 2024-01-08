#!/bin/bash
cd ..

# docker build -t discover304/real:1.1 ../docker
local_dir=/home/haobo/workspace # TODO modify this in need
container_dir=/workspace
docker run \
    --gpus all \
    --rm \
    -itd \
    --name real \
    -v $local_dir:$container_dir \
    -v /mnt/data/DataSet/S-KITTI/dataset:/mnt/data/DataSet/S-KITTI/dataset \
    --shm-size 32G \
    --ipc=host \
     discover304/real:1.1
