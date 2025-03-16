from fastcore.all import *



def main():
    path = Path(__file__).parent.parent.parent / "example_training_config.py"
    import os
    os.system(f"cp {path} .")