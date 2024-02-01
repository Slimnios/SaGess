
os=$(uname)
gpu_flag=""

if [ "$os" != "Darwin" ]; then
    # if not on Mac, include --gpus all
    gpu_flag="--gpus all"
fi

docker run  --rm -it ${gpu_flag} --shm-size=200M \
            -v "$(pwd):/root/workspace" \
            sagess_ws:latest

