conda create -n hypersloth -f environment.yml -y
pip install uv poetry
uv pip install -e ./
