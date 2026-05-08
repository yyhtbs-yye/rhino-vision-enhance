import types
from rhtrain.rhino_train import main

args_dict = {
    'config': 'configs/psrt/train_reds_x4_on_hat.yaml',
    'resume_from': '/home/yyh/python_workspaces/rhino-opsr/work_dirs/hat_psrt_reds_32_256/run_38/last.pt',
}

args = types.SimpleNamespace(**args_dict)

if __name__ == "__main__":

    main(args)