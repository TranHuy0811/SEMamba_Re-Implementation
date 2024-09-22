from pesq import pesq
from joblib import Parallel,delayed
import numpy as np
import torch

def compute_pesq_score(noisy_audio, clean_audio, rate=16000):
    try: 
        pesq_score = pesq(rate, clean_audio, noisy_audio, 'wb')
    except: 
        pesq_score = -1 # There would be some cases when it return an error
    return pesq_score



def compute_batch_pesq_percent(noisy_audios, clean_audios, cfg, rate=16000):
    num_workers = cfg['env_config']['num_workers']
    batch_pesq_score = Parallel(n_jobs=num_workers)(delayed(compute_pesq_score)(n,c) for n,c in zip(noisy_audios, clean_audios))
    batch_pesq_score = np.array(batch_pesq_score)
    
    if -1 in batch_pesq_score : 
        return None
    
    batch_pesq_score_percent = (batch_pesq_score - 1.0) / 3.5
    return torch.FloatTensor(batch_pesq_score_percent)




# PESQ function used in Evaluation process . Works the same as above
def eval_compute_pesq_score(noisy_audios, clean_audios, cfg, rate=16000):
    num_workers = cfg['env_config']['num_workers']
    eval_total_pesq_score = Parallel(n_jobs=num_workers)(delayed(compute_pesq_score)(
        n.squeeze().cpu().numpy(),
        c.squeeze().cpu().numpy()
    ) for n,c in zip(noisy_audios, clean_audios))

    eval_pesq_score = np.mean(eval_total_pesq_score)
    return eval_pesq_score
