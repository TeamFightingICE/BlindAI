# Deep Reinforcement Learning Blind AI

This page contains the source code and model of our deep reinforcement learning blind AI (Blind AI), the details of which are given in this [paper](https://arxiv.org/abs/2205.07444).

## Quickstart with Docker
- Boot DareFightingICE with the option `--limithp 400 400 --pyftg-mode --non-delay 0`.
- Run the docker container
```
docker run -it --rm --gpus all -v ${PWD}/ppo_pytorch:/app/ppo_pytorch -e SERVER_HOST=host.docker.internal ghcr.io/teamfightingice/blindai train --p2 MctsAi23i --encoder mel --id rnn_1_frame_256_mctsai23i --n-frame 1 --recurrent
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
- Boot DareFightingICE with the option `--limithp 400 400 --pyftg-mode`.
- Run the ```main.py``` file to train. e.g ```python main.py train --p2 MctsAi23i --encoder mel --id rnn_1_frame_256_mctsai23i --n-frame 1 --recurrent```
- Download the model from [here](https://drive.google.com/drive/folders/1xVIlMNNY8prY3HgMdPnqC37loaZUlwAJ?usp=sharing) and paste the folder named `trained_model` into the root directory.
- Run the ```python main.py test --encoder --p2 MctsAi23i --game_num number_of_games``` command to test the Blind AI.

## Model:
- [Click here.](https://drive.google.com/drive/folders/1xVIlMNNY8prY3HgMdPnqC37loaZUlwAJ?usp=sharing)<br>

## Command Description
- ```train``` is a command used to train Blind AI. Please run ```python main.py train --help``` for our explanation of the parameters.
- ```visualize``` is used to visualize the learning curve and calculate the area under the learning curve.
- ```analyze``` is used to calculate the win ratio and average HP difference between Blind AI and MctsAi23i.
- ```test``` is used to test the performance of the trained Blind AI. Please run ```python main.py test --help``` for our explanation of the parameters.

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
1. Software:
   - OS: Linux Ubuntu 20.04.3 LTS (Focal Fossa)
   - OpenJDK: 21.0.2
   - DareFightingICE: 6.3.1
   - Python: 3.12.3
   - pyftg: 2.3b0
2. Hardware:
   - CPU: Intel(R) Xeon(R) Gold 6258R CPU @ 2.70GHz
   - RAM: 188 GB
   - GPU: NVIDIA A100 80GB VRAM

## Performance against MctsAi23i
- Winning ratio: 0.54
- Average HP difference: 18.87   

## Deep learning libraries in use:
- pytorch 2.3.0
- torchaudio 2.3.0
- torchvision 0.18.0
