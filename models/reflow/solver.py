import os
import torch
import librosa
from tools.saver import Saver
from torch import autocast

def test(args, model, vocoder, loader_test, saver):
    model.eval()

    test_loss = 0.
    num_batches = len(loader_test)
    
    with torch.no_grad():
        for _, data in enumerate(loader_test):
            for k in data.keys():
                if not k.startswith('name'):
                    data[k] = data[k].to(args.device)
            print('>>', data['name'][0])

            # forward
            mel = model(data['units'], data['f0'], data['volume'], data['spk_id'], vocoder=vocoder, infer=True, return_wav=False, infer_step=args.infer.infer_step, method=args.infer.method)
            signal = vocoder.infer(mel, data['f0'])
           
            # loss
            loss = model(data['units'], data['f0'], data['volume'], data['spk_id'], vocoder=vocoder, gt_spec=data['mel'], infer=False)
            test_loss += loss.item()
            
            # log mel
            saver.log_spec(data['name'][0], data['mel'], mel)
            
            # log audio
            path_audio = os.path.join(args.data.valid_path, 'audio', data['name_ext'][0])
            audio, sr = librosa.load(path_audio, sr=args.data.sampling_rate)
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
            audio = torch.from_numpy(audio).unsqueeze(0).to(signal)
            saver.log_audio({data['name'][0] + '/gt.wav': audio, data['name'][0] + '/pred.wav': signal})
            
    test_loss /= num_batches 
    
    print('test_loss:', test_loss)
    return test_loss

def train(args, initial_global_step, model, optimizer, scheduler, vocoder, loader_train, loader_test):
    saver = Saver(args, initial_global_step=initial_global_step)
    
    num_batches = len(loader_train)
    start_epoch = initial_global_step // num_batches
    model.train()
    scaler = torch.cuda.amp.GradScaler()
    
    if args.train.amp_dtype == 'fp32':
        dtype = torch.float32
    elif args.train.amp_dtype == 'fp16':
        dtype = torch.float16
    elif args.train.amp_dtype == 'bf16':
        dtype = torch.bfloat16
    else:
        raise ValueError('Error: Unknown amp_dtype: ' + args.train.amp_dtype)
    for epoch in range(start_epoch, args.train.epochs):
        for batch_idx, data in enumerate(loader_train):
            saver.global_step_increment()
            optimizer.zero_grad()

            # unpack data
            for k in data.keys():
                if not k.startswith('name'):
                    data[k] = data[k].to(args.device)
            
            # forward
            if dtype == torch.float32:
                loss = model(data['units'].float(), data['f0'], data['volume'], data['spk_id'], aug_shift=data['aug_shift'], vocoder=vocoder, gt_spec=data['mel'].float(), infer=False)
            else:
                with autocast(device_type=args.device, dtype=dtype):
                    loss = model(data['units'], data['f0'], data['volume'], data['spk_id'], aug_shift=data['aug_shift'], vocoder=vocoder, gt_spec=data['mel'].float(), infer=False)
            
            # handle nan loss
            if torch.isnan(loss):
                raise ValueError('Error: NaN loss')
            else:
                # backpropagate
                if dtype == torch.float32:
                    loss.backward()
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                scheduler.step()
                
            # log loss
            if saver.global_step % args.train.interval_log == 0:
                current_lr =  optimizer.param_groups[0]['lr']
                print('epoch: {} | {:3d}/{:3d} | {} | batch/s: {:.2f} | lr: {:.6} | loss: {:.4f} | time: {} | step: {}'.format(
                        epoch, batch_idx, num_batches, args.env.expdir, args.train.interval_log / saver.get_interval_time(), current_lr, loss.item(), saver.get_total_time(), saver.global_step))
                
                saver.log_value({'train/loss': loss.item(), 'train/lr': current_lr})
            
            # validation
            if saver.global_step % args.train.interval_val == 0:
                optimizer_save = optimizer if args.train.save_opt else None
                
                # save latest
                saver.save_model(model, optimizer_save, postfix=f'{saver.global_step}')
                last_val_step = saver.global_step - args.train.interval_val
                if last_val_step % args.train.interval_force_save != 0:
                    saver.delete_model(postfix=f'{last_val_step}')
                
                test_loss = test(args, model, vocoder, loader_test, saver)
                    
                saver.log_value({'validation/loss': test_loss})      
                model.train()