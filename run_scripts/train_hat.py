import types
from rhtrain.rhino_train import main

args_dict = {
    'config': 'configs/hat/train_ffhq_32_256.yaml',
    'resume_from': 'work_dirs/baked_hat_ffhq_32_256/run_1/last.pt',
}

args = types.SimpleNamespace(**args_dict)

if __name__ == "__main__":

    main(args)