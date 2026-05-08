import types
from rhtrain.rhino_train import main

args_dict = {
    'config': 'configs/hat_gan/train_df2k_4x.yaml',
    'resume_from': 'work_dirs/hat_df2k_x4/run_7/last.pt',
}

args = types.SimpleNamespace(**args_dict)

if __name__ == "__main__":

    main(args)