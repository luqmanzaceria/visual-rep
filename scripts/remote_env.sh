# Environment for the shared 4090 server (source this). Everything lives on the network drive: the home disk is full.
export PATH=/mnt/luq/envs/vrep/bin:$PATH
export TORCH_HOME=/mnt/luq/cache/torch
export HF_HOME=/mnt/luq/cache/hf
export HF_HUB_OFFLINE=1          # huggingface.co is unreachable from the server; models are pre-copied
export TRANSFORMERS_OFFLINE=1
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
cd /mnt/luq/vrep
