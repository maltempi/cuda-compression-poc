# ZFP report

## Singularity env

### Pulling container image
```sh
singularity pull docker://maltempi/awave-dev:ubuntu20.04-cuda11.2-customcusz-zfp
```

### Building & running
```sh
# Replace ${image_sif_path} by pulled image path. (.sif)
$IMAGE=${image_sif_path}
mkdir build && cd build
singularity exec $IMAGE cmake ..
singularity exec $IMAGE make

singularity exec $IMAGE ./ex-api
```

## Ogbon env
```
./run-ogbon.sh
```

## Docker
```sh
docker run --rm -it --gpus all --env NVIDIA_DISABLE_REQUIRE=1 -v ${PWD}:/root/zfp maltempi/awave-dev:ubuntu20.04-cuda11.2-customcusz-zfp /bin/bash

cd ~/zfp
mkdir -p build && cd build
cmake ..
make
./ex-api

## With nsys
nsys profile --trace='cuda,nvtx,osrt' --cuda-memory-usage='true' ./ex-api
```