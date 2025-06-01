# conda create -n hypersloth -f environment.yml -y
pip install uv poetry
uv pip install torch
uv pip install transformers
pip install unsloth
uv pip install rich
uv pip install -e ./
