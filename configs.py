configs = {
    "sample_rate": 16000,
    "device": "cuda",
    "audio_len": 2,
    "lr": 0.0001,
    "batch_size": 64,
    "log_dir": "../results",
    "ckpt_path": "/home/rshahbazyan/Desktop/DL/results/Hubert/checkpoints/11736.pt",
    "log_steps": 1,
    "is_hubert": True,
    "model_params": {
        "mel_dim": 80,
        "dropout_rate": 0.1,
        "output_size": 50,
        "seq_len": 126
    }
}
