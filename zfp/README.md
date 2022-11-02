# ZFP report

## Singularity env

### Pulling container image
```
singularity pull docker://maltempi/awave-dev:ubuntu20.04-cuda11.2-customcusz-zfp
```

### Building & running
```
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