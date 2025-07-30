# Megatron MoE Minimal Testing example for Mixtral-8x22B

> For the complete guide please visit https://github.com/yanring/Megatron-MoE-ModelZoo

Note that this is a minimal example with little to no instructions. Use at your own risk.

## Prerequisites

This repo assumes that you already have a properly configured slurm environment with 128 H200/H100 GPUs.  

## Container Setup

Download cudnn version `9.11.0` with
```bash
wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.11.0.98_cuda12-archive.tar.xz
tar -xvJf cudnn-linux-x86_64-9.11.0.98_cuda12-archive.tar.xz
```

Build the container with docker
```bash
docker build --target base -f dockers/Dockerfile --tag mg-modelzoo_pytorch:25.06 --network host .
```

sqsh the container with enroot - this will create an image named `mg-modelzoo_pytorch+25.06.sqsh` in the current directory
```bash
enroot import dockerd://mg-modelzoo_pytorch:25.06
```
## Getting Megatron-LM

Clone the Megatron-LM repository
```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
```

checkout commit `1065e416cbe71d937d427ccd9360d627d6f82128`
```bash
cd Megatron-LM/ && git checkout 1065e416cbe71d937d427ccd9360d627d6f82128 && cd ../
```

## Create checkpoint

Install uv for package management:
```bash
pip install uv
```

Download `Mixtral-8x22B` checkpoint from HF:
```bash
 HF_TOKEN=<HF_TOKEN_HERE> uv run --with huggingface_hub ./ckpt_convert_scripts/Mixtral-8x22B/download_mixtral.py
```
## Convert to legacy Megatron-LM checkpoint

To convert the checkpoint we're going to re-use the container we've created earlier with pytorch already installed
```bash
srun --gpus-per-node=8 --container-image=./mg-modelzoo_pytorch+25.06.sqsh --container-mounts=$(pwd) --container-workdir=$(pwd) --pty bash
```

In the container we run:
```bash
export WORKDIR=$(pwd)

pip install accelerate==1.9.0 # dependency required 

cd Megatron-LM/ &&  git checkout 64b5ce94734d2938f513530ae52640c94fc4e7cf && cd ../ # convert needs this version of Megatron-LM

bash ./ckpt_convert_scripts/Mixtral-8x22B/convert_mixtral.sh

cd Megatron-LM/ &&  git checkout 1065e416cbe71d937d427ccd9360d627d6f82128 && cd ../ # afterwards we'll change back to 1065e416cbe71d937d427ccd9360d627d6f82128
```

Once you're done close the session with:
```bash
exit
```

## Start the training

> set your `WANDB_API_KEY` in `sbatch_nebius.slurm` before running

Run the slurm file
```bash
bash ./sbatch_nebius.slurm
```