# generative-adversarial-attacks


We propose generative adversarial attacks for different time-series models that generate a adversarial perturbation using a trained model. 
Using different combinations of architectures, we trained attacking models on pre-trained classifiers(which we call *learning-target*) and compared them with classical FGSM and iFGSM attacks.
The proposed adversarial attack with the model-based method makes attacks faster than classic. Our claims are supported by benchmarking via datasets from UCR and different models: recurrent, convolution, state-space, and transformers. Our method also works in black-box conditions when there is no access to the gradients of the attacked classifier(which we call *infernce-target*). 
Also, in analogy to iFGSM and diffusion models, an iterative generative attack has been proposed that attacks data sequentially.

## Data
You need to download [UCR dataset](https://paperswithcode.com/dataset/ucr-time-series-classification-archive) dataset and put it into the `datasets` directory.

## Quick Start

<!-- If you want to work with project with Docker, you can use folder **docker_scripts**.
Firstly copy file `credentials_example` to `credentials` and tune it with your variables for running docker. After you need to make docker image using command:

```
cd docker_scripts
bash build
```

For creating container run:

```
bash launch_container
``` -->

All the requirements are listed in `requirements.txt`
For install all packages run

```
pip install -r requirements.txt
```
To deploy the project and run experiments, you can check out the demo notebooks in the folder `notebooks`.
<!-- After you need to create folders `checkpoints` for saving classifier weights and `results` for saving adversarial attacks results. -->

<!-- Where are three basic steps: train classifier, attack model, train discriminator.
To run these steps you need to rename folder `config_examples` to `config`, then change assosiated config files in "config" folder and after that run assosiated python scrits `train_classifier.py`, `attack_run.py` and `train_discriminator.py`. -->

<!-- For example:
```
python train_classifier.py
``` -->
## Describtion

The goal of the project is to create generative adversarial attacks for time-series models, which could work faster even in the absence of access to the gradients of the attacked model.

## Content

| File or Folder | Content |
| --- | --- |
| checkpoints| folders for saving weights of the models |
| config | folder contains config files with params of models and paths |
| datasets | folder contains datasets
| notebooks | folder with notebooks for data visualisation and small experiments|
| source | folder with code|

<!-- ## Configs structure
We use a hydra-based experiment configuration structure. To run `train_discriminator.py` and `attack_run.py` you need to set the following configs:
* `- dataset:` choose the dataset you need from the `config/dataset/` directory.
* `- model:` choose the model for discriminator from the `config/model/` directory.
* `- model@attack_model:` choose the classifier model from the `config/model/` directory which was trained using a script `train_classifier.py`.
* `- model@disc_model_check:` select the model for discriminator from the `config/model/` directory if you want to get metrics of another  trained discriminator. Note that if you want to use it, set `use_disc_check: True` 
* `- attack_disc@attack:` choose the type of adversarial attack from the `config/attack/` directory. -->

