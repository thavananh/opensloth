# conda create -n opensloth -f environment.yaml -y

# conda %                                                                                                                            
conda create --name unsloth_env \
    python=3.11 \
    pytorch-cuda=12.1 \
    pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \
    -y
conda activate unsloth_env

pip install uv poetry
uv pip install torch transformers -U
# uv pip install 
pip install unsloth --no-deps
uv pip install rich
uv pip install -e ./

pip install -U xformers --index-url https://download.pytorch.org/whl/cu128
