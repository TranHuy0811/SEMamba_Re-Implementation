import torch.optim as optim
from torch.utils.data import DistributedSampler, DataLoader



def setup_optimizers(models, cfg):
    generator,discriminator = models
    learning_rate = cfg['training_config']['learning_rate']
    betas = (cfg['training_config']['adam_b1'],cfg['training_config']['adam_b2'])
    
    optim_g = optim.AdamW(generator.parameters(),lr=learning_rate,betas=betas)
    optim_d = optim.AdamW(discriminator.parameters(),lr=learning_rate,betas=betas)
    
    return optim_g, optim_d




def setup_schedulers(optimizers, cfg, last_epoch=-1):
    optim_g, optim_d = optimizers
    lr_decay = cfg['training_config']['lr_decay']
    
    scheduler_g = optim.lr_scheduler.ExponentialLR(optim_g, gamma=lr_decay, last_epoch=last_epoch)
    scheduler_d = optim.lr_scheduler.ExponentialLR(optim_d, gamma=lr_decay, last_epoch=last_epoch)
    
    return scheduler_g, scheduler_d
    



def create_dataloaders(dataset, cfg, train=True):
    if cfg['env_config']['num_gpus'] > 1:
        sampler = DistributedSampler(dataset)
        sampler.set_epoch(cfg['training_config']['training_epochs'])
        batch_size = (cfg['training_config']['batch_size'] // cfg['env_config']['num_gpus']) if train else 1
    else:
        sampler=None
        batch_size = cfg['training_config']['batch_size'] if train else 1
    num_workers = cfg['env_config']['num_workers'] if train else 1
    
    return DataLoader(
        dataset,
        num_workers=num_workers,
        shuffle=(sampler is None) and train,
        sampler=sampler,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=True if train else False
    )