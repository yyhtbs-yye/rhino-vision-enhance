import types
from rhtrain.rhino_train import main

args_dict = {
    'config': 'configs/hat/train_hat_ffhq_32_256_tiny.yaml',
    'resume_from': None,
}

args = types.SimpleNamespace(**args_dict)

if __name__ == "__main__":

    main(args)