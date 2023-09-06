docker container create -it --privileged \
    --name sd-v1.4-cpu-$USER-dev \
    --ulimit memlock=-1:-1 --net=host --cap-add=IPC_LOCK \
    --ipc=host \
    -v $(readlink -f `pwd`):/workspace \
    --workdir /workspace \
    --cpus=8 \
    sd-v1.4-cpu:latest

