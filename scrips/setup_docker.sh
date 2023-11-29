#!/bin/bash
cd /home/haobo/workspace/realtime
docker build -t real:1.0 docker
local_dir=/home/haobo/workspace # TODO modify this in need
container_dir=/workspace
docker run --gpus all --rm -itd --name real -v $local_dir:$container_dir --shm-size 32G --ipc=host real:1.0
docker exec real bash /workspace/realtime/setup.sh