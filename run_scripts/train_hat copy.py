import types
from rhtrain.rhino_train import main

args_dict = {
    'config': 'configs/psrt/train_reds_x4_on_hat.yaml',
    'resume_from': '
}

args = types.SimpleNamespace(**args_dict)

if __name__ == "__main__":

    main(args)