#!/bin/bash

set -e

source ~/miniconda3/etc/profile.d/conda.sh && conda activate extend

# start ui
streamlit run extend/demo/ui.py experiments/extend-longformer-large/2021-10-22/09-11-39/checkpoints/best.ckpt --server.port 22001 &

# start serve
PYTHONPATH=$(pwd) python extend/demo/serve.py experiments/extend-longformer-large/2021-10-22/09-11-39/checkpoints/best.ckpt -p 22002 &

# Wait for any process to exit
wait -n

# Exit with status of process that exited first
exit $?