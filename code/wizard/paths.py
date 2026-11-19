"""Where the data and the base model live. Set through the environment:

    WIZARD_DATA   root of the pre-encoded training data (the folders windows/, samples/, roam_memory/, scenefly/, ...)
    WIZARD_WAN    the official Wan2.2-TI2V-5B directory (DiT shards, Wan2.2_VAE.pth, the UMT5 encoder and tokenizer)
"""
import os

DATA_ROOT = os.environ.get("WIZARD_DATA", "data")
WAN_DIR = os.environ.get("WIZARD_WAN", "checkpoints/Wan2.2-TI2V-5B")
