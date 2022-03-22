#!/bin/bash

set -e

# setup conda
source ~/miniconda3/etc/profile.d/conda.sh

## create conda env
read -rp "Enter environment name: " env_name
read -rp "Enter python version (e.g. >=3.8): " python_version
conda create -yn "$env_name" python="$python_version"
conda activate "$env_name"

# install torch
echo "Not sure which CUDA version you have? Check out https://stackoverflow.com/a/68499241/1908499"
read -rp "Enter cuda version (10.2, 11.1 or none to avoid installing cuda support): " cuda_version
if [[ "$cuda_version" == "none" ]]; then
    conda install -y pytorch=1.9.0 torchvision cpuonly -c pytorch
elif [[ "$cuda_version" == "10.2" || "$cuda_version" == "11.1" ]]; then
    conda install -y pytorch=1.9.0 torchvision cudatoolkit=$cuda_version -c pytorch -c conda-forge
else
    echo "Expected values for cuda_version are {none, 10.2, 11.1}, but found \"$cuda_version\""
    exit 1
fi

# install python requirements
pip install -r requirements.txt
pip install classy-core
classy --install-autocomplete

# install repo
pip install -e .

# download spacy model
python -m spacy download en_core_web_sm

echo "Classy successfully installed. Don't forget to activate your environment!"
echo "$> conda activate ${env_name}"
