# My Re-Implementation of SEMamba Model


## Requirement
Install needed dependencies, See `requirements.txt` for more details

```setup
pip install -r requirements.txt
```


## Data Preparation
I'm using a txt file containing a list of noisy and clean audio separated by a "\t" character.

```example
"path_to_noisy_audio"	"path_to_clean_audio"
```


## Train
```train
python train.py\
  --export_path\ 	(Your export folder, default: 'export')
  --config\		(Config file, default: 'config.yaml')
  --dis_pretrained\ 	(Discriminator model pretrained, default: None)
  --gen_pretrained\ 	(Generator model pretrained, default: None)
  --optimizer\ 		(Your last optimizer state, default: None)
  --scheduler\ 		(Your last learning rate and step, default: None)
  --wandb_key 		(This is for visualization, Access this site (https://wandb.ai/authorize) to get your API Key)
```


## Inference
I do provide a pre-trained model for inference, stored in "pretrained" folder
```infer
python inference.py\
  --input_file 		(Noisy Audio Path)
  --gen_pretrained 	(Path to pre-trained model)
  --config 		(Path to config file, default: 'config.yaml')
  --export_path 	(Exported folder, default: 'infer_result')
