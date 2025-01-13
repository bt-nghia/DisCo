conf = {
    # "n_dim": 16,
    # "n_head": 2,
    # "n_layer": 2,
    # "n_token": 4490, # number of item + [eos, sos, pad]
    "seq_len": 10,  # token len/number of item pass to encoder
    "pad": 0,
    "sos": 1,
    "eos": 2,
    # "batch_size": 16,
    # "beam_width": 3,  # perform multiple bundle generation
    "ckpt_path": "/checkpoints",
    "pseudo_bundle_file": "pseudo_bundle.npy",
    "data_path": "datasets",
    "epochs": 100,
    "p_epochs": 10,
    "beta": 1e-6
}
