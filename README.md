# Deep Reinforcement Learning Blind AI

This page contains the source code and model of our deep reinforcement learning blind AI (Blind AI), the details of which are given in this [paper](https://arxiv.org/abs/2205.07444).

## Quickstart with Docker
- Boot DareFightingICE with the option `--limithp 400 400 --pyftg-mode --non-delay 0`.
- Run the docker container
```
docker run -it --rm --gpus all -v ${PWD}/ppo_pytorch:/app/ppo_pytorch -e SERVER_HOST=host.docker.internal ghcr.io/teamfightingice/blindai train --p2 MctsAi23i --encoder mel --id rnn_1_frame_256_mctsai23i --n_frame 1 --recurrent
```

## Installation:
- Install miniconda: https://docs.conda.io/en/latest/miniconda.html.
- Clone the repo: `git clone https://github.com/TeamFightingICE/FightingICE`.
- Create and activate conda env:
```
git clone https://github.com/TeamFightingICE/BlindAI
cd BlindAI
conda env create -n ice -f environment.yml
conda activate ice
```
- Boot DareFightingICE with the option `--limithp 400 400 --grpc-auto --non-delay 0`.
- Run the ```train.py``` file to train. e.g ```python main.py train --p2 MctsAi23i --encoder mel --id rnn_1_frame_256_mctsai23i --n-frame 1 --recurrent```
- Download the model from [here](https://drive.google.com/file/d/1Kz_qzUmcJOAj0B9JfFbTJ1FzRFu8fg0B/view?usp=share_link) and paste the folder named `trained_model` into the root directory.
- Run the ```trained_ai/test.py``` to test the Blind AI. e.g. ```TBU```

## Model:
- [Click here.](https://drive.google.com/file/d/1Kz_qzUmcJOAj0B9JfFbTJ1FzRFu8fg0B/view?usp=share_link)<br>

## Command Description
- ```train``` is a command used to train Blind AI. Please run ```python main.py train --help``` for our explanation of the parameters.
- ```visualize``` is used to visualize the learning curve and calculate the area under the learning curve.
- ```analyze``` is used to calculate the win ratio and average HP difference between Blind AI and MctsAi23i.

## Get sound design evaluation metrics
- After finishing your sound design, please run the following command to train Blind AI:
  ```python train.py --p2 MctsAi23i --encoder fft --id {experiment_id} --n_frame 1 --recurrent```, where you can decide ```experiment_id``` on your own
- After training, a result file with the name ```result_fft_{experiment_id}_rnn.txt``` is created. Please run ```visualize``` as follows: ```python main.py visualize --file result_fft_{experiment_id}_rnn.txt --title FFT```. A plot will be shown and the area under the learning curve will be printed out.
- Before testing the performance of the Blind AI against MctsAi23i, please remove all the files under ```log/point``` of DareFightingICE.
- Please revise the ```path``` parameter of the Blind AI in line 16 of ```trained_ai/test.py``` to your trained model location.
- Run ```TBU``` to begin testing.
- After testing, please run ```python main.py analyze --path {path}``` where ```path``` is the location of ```log/point``` of DareFightingICE.
- Both win ratio and average HP difference will be printed out.

## Tested Environment
- Intel(R) Xeon(R) W-2135 CPU @ 3.70GHz   3.70 GHz
- 16.0 GB RAM
- NVIDIA Quadro P1000 GPU
- Windows 10 Pro
- Python 3.12
- DareFightingICE 7.0

## Performance against MctsAi23i
- Winning ratio: 0.54
- Average HP difference: 18.87   

## Deep learning libraries in use:
- pytorch 2.3.0
- torchaudio 2.3.0
- torchvision 0.18.0
