import os
import argparse
import torch
import torchaudio
from util import load_config, LoadPretrainedPath
from models.stft import STFT
from models.istft import Inverse_STFT
from models.generator_model import SEMamba_Advanced

device = None


def Inference(args, cfg):
    device = torch.device("cuda")
    n_fft, hop_size, win_size = cfg['audio_n_stft_config']['n_fft'], cfg['audio_n_stft_config']['hop_size'], cfg['audio_n_stft_config']['win_size']
    sample_rate = cfg['audio_n_stft_config']['sample_rate']
    compress_factor = cfg['model_config']['compress_factor']

    noisy_audio, initial_rate = torchaudio.load(args.input_file)
    if initial_rate != sample_rate:
        noisy_audio = torchaudio.transforms.Resample(orig_freq = initial_rate, new_freq = sample_rate)(noisy_audio)

    gen_model = SEMamba_Advanced(cfg).to(device)
    gen_state_dict = torch.load(args.gen_pretrained, map_location=device)
    gen_model.load_state_dict(gen_state_dict, strict=False)
    gen_model.eval()

    with torch.no_grad():
        norm_factor = torch.sqrt(len(noisy_audio) / torch.sum(noisy_audio ** 2.0))
        noisy_audio = noisy_audio * norm_factor
        noisy_mag, noisy_pha, _ = STFT(noisy_audio, n_fft, hop_size, win_size, compress_factor)
        mag_gen, pha_gen, _ = gen_model(noisy_mag.to(device), noisy_pha.to(device))
        audio_gen = Inverse_STFT(mag_gen, pha_gen, n_fft, hop_size, win_size, compress_factor)
        audio_gen = audio_gen / norm_factor

    file_name = os.path.basename(args.input_file)
    torchaudio.save(os.path.join(args.export_path, file_name), audio_gen.cpu(), sample_rate)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--gen_pretrained", default=None)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--export_path", default="infer_result")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg['pretrained_config']['gen_pretrained'] = LoadPretrainedPath(args.gen_pretrained, 'gen_pretrained', cfg)

    if not torch.cuda.is_available():
        raise RuntimeError("CPU is not supported right now !")

    # Create export folder if not exist
    os.makedirs(args.export_path, exist_ok=True)

    Inference(args, cfg)


if __name__ == "__main__":
    main()
