import argparse
import torch
from torch.optim import lr_scheduler
from tools import utils
from models.reflow.data_loaders import get_data_loaders
from models.reflow.vocoder import Vocoder
from models.reflow.main import Unit2Wav_VAE
from models.reflow.solver import train

def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default=r'./configs/reflow-vae-wavenet.yaml')
    return parser.parse_args(args=args, namespace=namespace)

if __name__ == '__main__':
    cmd = parse_args()

    args = utils.load_config_yaml(cmd.config)
    
    # load vocoder
    vocoder = Vocoder(args.vocoder.type, args.vocoder.ckpt)
    
    # load model            
    if args.model.type == 'RectifiedFlow_VAE':
        model = Unit2Wav_VAE(
                    args.data.encoder_out_channels,
                    args.model.n_spk,
                    args.model.use_pitch_aug,
                    vocoder.dimension,
                    args.model.n_layers,
                    args.model.n_chans,
                    args.model.n_hidden,
                    args.model.back_bone,
                    args.model.use_attention
                    )    
    else:
        raise ValueError(f" [x] Unknown Model: {args.model.type}")
    
    # load parameters
    optimizer = torch.optim.AdamW(model.parameters())
    initial_global_step, model, optimizer = utils.load_model(args.env.expdir, model, optimizer, device=args.device)
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = args.train.lr
        param_group['lr'] = args.train.lr * args.train.gamma ** max((initial_global_step - 2) // args.train.decay_step, 0)
        param_group['weight_decay'] = args.train.weight_decay
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.train.decay_step, gamma=args.train.gamma, last_epoch=initial_global_step-2)
    
    # device
    if args.device == 'cuda':
        torch.cuda.set_device(args.env.gpu_id)
    model.to(args.device)
    
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(args.device)
                    
    # datas
    loader_train, loader_valid = get_data_loaders(args, whole_audio=False)
    
    # run
    train(args, initial_global_step, model, optimizer, scheduler, vocoder, loader_train, loader_valid)