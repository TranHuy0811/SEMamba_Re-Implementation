import os
import argparse
import warnings
import time
import wandb
import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
import torch.nn.functional as F

from util import (
    load_config, initialize_seed, build_env, 
    initialize_process_group, log_model_info,
    LoadPretrainedPath
)

from setup import (
    create_dataloaders,setup_optimizers,setup_schedulers
)

from models.pesq_compute import (
    compute_batch_pesq_percent, compute_pesq_score, eval_compute_pesq_score
)

from dataloaders.create_dataset import CreateDataset
from models.phase_loss import phase_losses
from models.stft import STFT
from models.istft import Inverse_STFT
from models.generator_model import SEMamba_Advanced
from models.discriminator_model import Discriminator


torch.backends.cudnn.benchmark=True
warnings.simplefilter(action='ignore', category=FutureWarning)




def Train(rank,args,cfg):
    num_gpus = cfg['env_config']['num_gpus']
    n_fft, hop_size, win_size = cfg['audio_n_stft_config']['n_fft'], cfg['audio_n_stft_config']['hop_size'], cfg['audio_n_stft_config']['win_size']
    compress_factor = cfg['model_config']['compress_factor']
    batch_size = cfg['training_config']['batch_size'] // cfg['env_config']['num_gpus']

    if num_gpus >= 1:
        initialize_process_group(cfg, rank)
        device = torch.device('cuda:{:d}'.format(rank))
    else:
        raise RuntimeError("Mamba needs GPU acceleration")

    
    
    # ------------------------------------------------------- #
    
    
    
    ''' Load Model, Optim and Scheduler '''
    
    generator = SEMamba_Advanced(cfg).to(device)
    discriminator = Discriminator().to(device)

    if rank == 0 : log_model_info(generator, args.export_path)
    
    optim_g, optim_d = setup_optimizers((generator, discriminator), cfg)
    
    
    # Loading from pretrained
    
    if cfg['pretrained_config']['gen_pretrained'] != None:
        cur_path = cfg['pretrained_config']['gen_pretrained']
        print('Loading generator pretrained from {}'.format(cur_path))
        state_dict_g = torch.load(cur_path, map_location=device)
        generator.load_state_dict(state_dict_g, strict=False)
        
        
    if cfg['pretrained_config']['dis_pretrained'] != None:
        cur_path = cfg['pretrained_config']['dis_pretrained']
        print('Loading discriminator pretrained from {}'.format(cur_path))
        state_dict_d = torch.load(cur_path, map_location=device)
        discriminator.load_state_dict(state_dict_d, strict=False)
    
    
    if cfg['pretrained_config']['optimizer'] != None:
        cur_path = cfg['pretrained_config']['optimizer']
        print('Loading optimizer from {}'.format(cur_path))
        state_dict_optim = torch.load(cur_path, map_location=device)
        optim_g.load_state_dict(state_dict_optim['optim_g'])
        optim_d.load_state_dict(state_dict_optim['optim_d'])
        
        
    step = -1
    last_epoch = -1
    if cfg['pretrained_config']['scheduler'] != None:
        cur_path = cfg['pretrained_config']['scheduler']
        print('Loading scheduler last epoch from {}'.format(cur_path))
        state_dict_scheduler = torch.load(cur_path, map_location=device)
        last_epoch = state_dict_scheduler['last_epoch']
        step = state_dict_scheduler['last_step']
        
    scheduler_g, scheduler_d = setup_schedulers((optim_g, optim_d), cfg, last_epoch) # Loading Scheduler
        
        
    
    # Enable DistributedDataParallel
    if num_gpus > 1 and torch.cuda.is_available():
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        discriminator = DistributedDataParallel(discriminator, device_ids=[rank]).to(device)

        
    
    ''' Dataset Creation '''

    train_dataset = CreateDataset(cfg['data_config']['train_dataset_path'], cfg, pcs=cfg['training_config']['use_pcs'], do_split=cfg['data_config']['do_split'])
    train_dataloader = create_dataloaders(train_dataset, cfg, train=True)
    if rank == 0:
        eval_dataset = CreateDataset(cfg['data_config']['eval_dataset_path'], cfg, pcs=False, do_split=False)
        eval_dataloader = create_dataloaders(eval_dataset, cfg, train=False)


    ''' Initialize Wandb for Logging ''' 
    if rank == 0: 
        wandb.init(project="SEMamba_Advanced")



    # ------------------------------------------------------- #    

    
    
    ''' Training '''
    
    generator.train()
    discriminator.train()


    for epoch in range(cfg['training_config']['training_epochs']):
        for batch in train_dataloader:
            step += 1
            clean_audio, noisy_mag, noisy_pha, noisy_com, clean_mag, clean_pha, clean_com = batch
            clean_audio = clean_audio.to(device, non_blocking=True)
            noisy_mag = noisy_mag.to(device, non_blocking=True)
            noisy_pha = noisy_pha.to(device, non_blocking=True)
            clean_mag = clean_mag.to(device, non_blocking=True)
            clean_pha = clean_pha.to(device, non_blocking=True)
            clean_com = clean_com.to(device, non_blocking=True)

            ones = torch.ones(batch_size).to(device, non_blocking=True)

            mag_gen, pha_gen, com_gen = generator(noisy_mag, noisy_pha)
            audio_gen = Inverse_STFT(mag_gen, pha_gen, n_fft, hop_size, win_size, compress_factor)


            ''' Discriminator '''
            optim_d.zero_grad()

            metric_r = discriminator(clean_mag, clean_mag)
            metric_g = discriminator(clean_mag, mag_gen.detach())
            batch_pesq_score = compute_batch_pesq_percent(audio_gen.detach().cpu().numpy(), clean_audio.cpu().numpy(), cfg, rate=16000)

            dis_loss_r = F.mse_loss(ones, metric_r.flatten())

            if batch_pesq_score != None: 
                dis_loss_g = F.mse_loss(batch_pesq_score.to(device), metric_g.flatten())
            else:
                dis_loss_g = 0

            dis_total_loss = dis_loss_r + dis_loss_g

            dis_total_loss.backward()
            optim_d.step()


            ''' Generator '''
            optim_g.zero_grad()

            # Time Loss
            gen_loss_time = F.l1_loss(clean_audio, audio_gen)

            # Magnitude Loss
            gen_loss_mag  = F.mse_loss(clean_mag, mag_gen)

            # Complex Loss
            gen_loss_com = F.mse_loss(clean_com, com_gen) * 2
            
            # Metric Loss
            metric_g = discriminator(clean_mag, mag_gen)
            gen_loss_metric = F.mse_loss(ones, metric_g.flatten())

            # Consistancy Loss
            _, _, rep_com = STFT(audio_gen, n_fft, hop_size, win_size, compress_factor, addeps=True)
            gen_loss_con = F.mse_loss(com_gen, rep_com) * 2

            # Anti-wrapping Phase Loss
            loss_ip, loss_gd, loss_iaf = phase_losses(clean_pha, pha_gen, cfg)
            gen_loss_pha = loss_ip + loss_gd + loss_iaf

            gen_total_loss = (
                gen_loss_time * cfg['training_config']['weighted_loss']['time'] +
                gen_loss_mag * cfg['training_config']['weighted_loss']['magnitude'] +
                gen_loss_com * cfg['training_config']['weighted_loss']['complex'] + 
                gen_loss_metric * cfg['training_config']['weighted_loss']['metric'] +
                gen_loss_con * cfg['training_config']['weighted_loss']['consistancy'] +
                gen_loss_pha * cfg['training_config']['weighted_loss']['phase']
            )

            gen_total_loss.backward()
            optim_g.step()

            
            # If NaN happend in training period, RaiseError
            if torch.isnan(gen_total_loss).any():
                raise ValueError("NaN values found in gen_total_loss")
            
            
            
            # ------------------------------------------------------- #
            


            ''' Logging and Saving'''
            if rank == 0: 

                # STDOUT Logging
                if step % cfg['env_config']['stdout_interval'] == 0:
                    with torch.no_grad():
                        time_error = gen_loss_time.item()
                        mag_error  = gen_loss_mag.item()
                        com_error = F.mse_loss(clean_com, com_gen).item()
                        metric_error = gen_loss_metric.item()
                        con_error = F.mse_loss(com_gen, rep_com)
                        pha_error = gen_loss_pha.item()
                        gen_total_error = gen_total_loss.item()
                        dis_total_error = dis_total_loss.item()
                        
                        print(
                            'Step : {:d}, Disc Loss : {:4.6f}, Gen Loss : {:4.6f}, Metric Loss: {:4.3f}, '
                            'Mag Loss: {:4.3f}, Pha Loss: {:4.3f}, Com Loss: {:4.3f}, Time Loss: {:4.3f}, Cons Loss: {:4.3f}'.format(
                                step, dis_total_error, gen_total_error, metric_error, mag_error, pha_error, com_error,
                                time_error, con_error
                            )
                        )
                
                
                # Saving Checkpoint
                if step % cfg['env_config']['checkpoint_interval'] == 0 and step != 0:
                    print(f"Saving Checkpoint at steps : {step}")
                    
                    exp_name = f"{args.checkpoint_path}/generator_{step:08d}.pth"
                    torch.save( (generator.module if num_gpus > 1 else generator).state_dict(), exp_name)
                    
                    exp_name = f"{args.checkpoint_path}/discriminator_{step:08d}.pth"
                    torch.save( (discriminator.module if num_gpus > 1 else discriminator).state_dict(), exp_name)
                    
                    exp_name = f"{args.checkpoint_path}/optimizer_{step:08d}.pth"
                    torch.save({
                        'optim_g': optim_g.state_dict(),
                        'optim_d': optim_d.state_dict()
                    }, exp_name)
                    
                    exp_name = f"{args.checkpoint_path}/scheduler_{step:08d}.pth"
                    torch.save({
                        'last_epoch': epoch,
                        'last_step': step
                    },exp_name)
                    
                
                # Summary Logging
                if step % cfg['env_config']['summary_interval'] == 0:
                    time_error = gen_loss_time.item()
                    mag_error  = gen_loss_mag.item()
                    com_error = F.mse_loss(clean_com, com_gen).item()
                    metric_error = gen_loss_metric.item()
                    con_error = F.mse_loss(com_gen, rep_com)
                    pha_error = gen_loss_pha.item()
                    gen_total_error = gen_total_loss.item()
                    dis_total_error = dis_total_loss.item()
                    log_pesq_score = torch.mean(batch_pesq_score * 3.5 + 1).item()
        
                    wandb.log({"Training/Discriminator Loss": dis_total_error}, step=step)
                    wandb.log({"Training/Generator Loss": gen_total_error}, step=step)
                    wandb.log({"Training/Metric Loss": metric_error}, step=step)
                    wandb.log({"Training/Magnitude Loss": mag_error}, step=step)
                    wandb.log({"Training/Phase Loss": pha_error}, step=step)
                    wandb.log({"Training/Complex Loss": com_error}, step=step)
                    wandb.log({"Training/Time Loss": time_error}, step=step)
                    wandb.log({"Training/Consistancy Loss": con_error}, step=step)
                    wandb.log({"Training/Training Pesq Score ": log_pesq_score}, step=step)
                
                # Evaluation
                if step % cfg['env_config']['evaluation_interval'] == 0 and step != 0:
                    generator.eval()
                    torch.cuda.empty_cache()
                    audios_r, audios_g = [], []
                    val_mag_err_total = 0
                    val_pha_err_total = 0
                    val_com_err_total = 0
                    with torch.no_grad():
                        for j,batch in enumerate(eval_dataloader):
                            clean_audio, noisy_mag, noisy_pha, noisy_com, clean_mag, clean_pha, clean_com = batch
                            clean_audio = clean_audio.to(device, non_blocking=True)
                            noisy_mag = noisy_mag.to(device, non_blocking=True)
                            noisy_pha = noisy_pha.to(device, non_blocking=True)
                            clean_mag = clean_mag.to(device, non_blocking=True)
                            clean_pha = clean_pha.to(device, non_blocking=True)
                            clean_com = clean_com.to(device, non_blocking=True)
                            
                            mag_gen, pha_gen, com_gen = generator(noisy_mag, noisy_pha)
                            audio_gen = Inverse_STFT(mag_gen, pha_gen, n_fft, hop_size, win_size, compress_factor)
                            
                            audios_r += torch.split(clean_audio, 1, dim=0) # [1, T] * B
                            audios_g += torch.split(audio_gen, 1, dim=0)
                            
                            val_mag_err_total += F.mse_loss(clean_mag, mag_gen).item()
                            val_ip_err, val_gd_err, val_iaf_err = phase_losses(clean_pha, pha_gen, cfg)
                            val_pha_err_total += (val_ip_err + val_gd_err + val_iaf_err).item()
                            val_com_err_total += F.mse_loss(clean_com, com_gen).item()
                            
                        val_mag_err = val_mag_err_total / (j+1)
                        val_pha_err = val_pha_err_total / (j+1)
                        val_com_err = val_com_err_total / (j+1)
                        val_pesq_score = eval_compute_pesq_score(audios_g, audios_r, cfg)
                        
                        print('Steps : {:d}, PESQ Score: {:4.3f}'.format(step, val_pesq_score))
                        wandb.log({"Validation/PESQ Score": val_pesq_score}, step=step)
                        wandb.log({"Validation/Magnitude Loss": val_mag_err}, step=step)
                        wandb.log({"Validation/Phase Loss": val_pha_err}, step=step)
                        wandb.log({"Validation/Complex Loss": val_com_err}, step=step)
                        
                    generator.train()
                    
            
        scheduler_g.step()
        scheduler_d.step()
            
        if rank == 0:
            print(f"Finished epoch {epoch}.")

    
    
    # Save model when finished training
    if rank == 0:
        print("Training Finished !")

        exp_name = f"{args.result_path}/generator_{step:08d}.pth"
        torch.save( (generator.module if num_gpus > 1 else generator).state_dict(), exp_name)
                        
        exp_name = f"{args.result_path}/discriminator_{step:08d}.pth"
        torch.save( (discriminator.module if num_gpus > 1 else discriminator).state_dict(), exp_name)
                        
        exp_name = f"{args.result_path}/optimizer_{step:08d}.pth"
        torch.save({
            'optim_g': optim_g.state_dict(),
            'optim_d': optim_d.state_dict()
        }, exp_name)
                        
        exp_name = f"{args.result_path}/scheduler_{step:08d}.pth"
        torch.save({
            'last_epoch': epoch,
            'last_step': step
        },exp_name)

            



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--export_path',default='export')
    parser.add_argument('--dis_pretrained',default=None)
    parser.add_argument('--gen_pretrained',default=None)
    parser.add_argument('--optimizer',default=None)
    parser.add_argument('--scheduler',default=None)
    parser.add_argument('--config',default='config.yaml')
    parser.add_argument('--wandb_key',default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    num_gpus = cfg['env_config']['num_gpus']
    available_gpus= torch.cuda.device_count()
    seed = cfg['env_config']['seed']

    cfg['pretrained_config']['dis_pretrained'] = LoadPretrainedPath(args.dis_pretrained, 'dis_pretrained', cfg)
    cfg['pretrained_config']['gen_pretrained'] = LoadPretrainedPath(args.gen_pretrained, 'gen_pretrained', cfg)
    cfg['pretrained_config']['optimizer'] = LoadPretrainedPath(args.optimizer, 'optimizer', cfg)
    cfg['pretrained_config']['scheduler'] = LoadPretrainedPath(args.scheduler, 'scheduler', cfg)


    if num_gpus > available_gpus :
        warnings.warn(
            f"Warning: The actual number of available GPUs ({available_gpus}) is less than the .yaml config ({num_gpus}). Auto reset to num_gpu = {available_gpus}",
            UserWarning
        )
        cfg['env_config']['num_gpus'] = available_gpus
        num_gpus = available_gpus
        time.sleep(5)

    args.checkpoint_path = os.path.join(args.export_path, "checkpoint")
    args.result_path = os.path.join(args.export_path, "result")

    initialize_seed(seed)
    build_env(args.config, 'config.yaml', args.export_path, args.checkpoint_path, args.result_path)


    if torch.cuda.is_available():
        num_available_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_available_gpus}")
    else:
        print("CUDA is not available.")

    wandb.login(key=args.wandb_key)
    if num_gpus > 1:
        mp.spawn(Train, nprocs=num_gpus, args=(args, cfg))
    else:
        Train(0,args,cfg)



if __name__ == '__main__':
    main()