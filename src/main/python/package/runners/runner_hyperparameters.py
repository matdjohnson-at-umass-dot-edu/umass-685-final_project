import torch

SETimesByT5Vaswani2017Kocmi2018_0 = {
    'dataset_transformer_name': 'dataset_transformer_setimesbyt5',
    'model_name': 'transformer_vaswani2017',
    'trainer_name': 'model_trainer_kocmi2018',
    # corresponds to dictionary 'get' calls in the dataset_loader constructor
    'dataset_transformer_hyperparameters': {},
    # corresponds to dictionary 'get' calls in the model constructor
    'model_hyperparameters': {
        'd_model': 512,
        'nhead': 8,
        # number of encoders is 3 times that of decoders, following Xue 2021 - ByT5 - Sec 3.1
        'num_encoder_layers': 9,
        'num_decoder_layers': 3,
        'dim_feedforward': 2048,
        'dropout': 0.1,
        'activation': torch.nn.functional.relu,
        'custom_encoder': None,
        'custom_decoder': None,
        'layer_norm_eps': 1e-5,
        'batch_first': True,
        'norm_first': False,
        'bias': True,
        'device': None,
        'dtype': None
    },
    # corresponds to dictionary 'get' calls in the trainer constructor
    'trainer_hyperparameters': {
        # optimization and lr schedule following Kocmi 2018 - Trivial TL - Sec 3
        'optimizer_name': 'Adam',
        'lr_scheduler_name': 'ExponentialLR',
        'initial_lr': 0.2,
        'epochs': 1,
        'batch_size': 10
    }
}
