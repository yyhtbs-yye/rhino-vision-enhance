import types
from rhtrain.rhino_train import main

args_dict = {
    'config': 'configs/basicvsrmm/train_reds000_64_256.yaml',
    'resume_from': 'work_dirs/basicvsrmm_reds_64_256/run_20/last.pt',
}

args = types.SimpleNamespace(**args_dict)

if __name__ == "__main__":
    main(args)