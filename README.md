# generative-adversarial-attacks


We propose generative adversarial attacks for different time-series models that generate an adversarial perturbation using a trained model. 
Using different combinations of architectures, we trained attacking models on pre-trained classifiers(which we call *learning-target*) and compared them with classical FGSM and iFGSM attacks.
The proposed adversarial attack with the model-based method makes attacks faster than classic. Our claims are supported by benchmarking via datasets from UCR and different models: recurrent, convolution, state-space, and transformers. Our method also works in black-box conditions when there is no access to the gradients of the attacked classifier(which we call *inference-target*). 
Also, in analogy to iFGSM and diffusion models, an iterative generative attack has been proposed that attacks data sequentially.

## Data
You need to download the [UCR dataset](https://paperswithcode.com/dataset/ucr-time-series-classification-archive) dataset and put it into the `datasets` directory.

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

## Results

<div style="overflow-x:auto">
<table>
  <thead>
    <tr>
      <th rowspan="2">Target model</th>
      <th rowspan="2">Attack</th>
      <th rowspan="2">Generative model</th>
      <th colspan="3">PowerCons</th>
      <th colspan="3">Strawberry</th>
    </tr>
    <tr>
      <th>FR↑</th><th>E↑</th><th>TA↓</th>
      <th>FR↑</th><th>E↑</th><th>TA↓</th>
    </tr>
  </thead>
  <tbody>
    <!-- LSTM -->
    <tr>
      <td rowspan="9">LSTM</td>
      <td>Unattacked</td><td>-</td>
      <td>0</td><td>0.044</td><td>0.957</td>
      <td>0</td><td>0.114</td><td>0.862</td>
    </tr>
    <tr><td>FGSM</td><td>-</td>
      <td>0.831</td><td><u>0.826</u></td><td><u>0.126</u></td>
      <td><u>0.634</u></td><td><u>0.574</u></td><td><u>0.349</u></td></tr>
    <tr><td>iFGSM</td><td>-</td>
      <td><u>0.832</u></td><td>0.814</td><td>0.126</td>
      <td><b><u>0.691</u></b></td><td><b><u>0.775</u></b></td><td><b><u>0.172</u></b></td></tr>
    <tr><td>Iterative Gen. Attack (our)</td><td>LSTM</td>
      <td>0.522</td><td>0.333</td><td>0.5</td>
      <td>0.24</td><td>0.24</td><td>0.715</td></tr>
    <tr><td>Iterative Gen. Attack (our)</td><td>ResCNN</td>
      <td>0.482</td><td>0.778</td><td>0.5</td>
      <td>0.363</td><td>0.329</td><td>0.639</td></tr>
    <tr><td>Iterative Gen. Attack (our)</td><td>PatchTST</td>
      <td><b><u>0.863</u></b></td><td><b><u>0.86</u></b></td><td><b><u>0.085</u></b></td>
      <td>0.501</td><td>0.371</td><td>0.484</td></tr>
    <tr><td>Gen. attack (our)</td><td>LSTM</td>
      <td>0.311</td><td>0.493</td><td>0.691</td>
      <td>0.303</td><td>0.217</td><td>0.681</td></tr>
    <tr><td>Gen. attack (our)</td><td>ResCNN</td>
      <td>0.718</td><td>0.6</td><td>0.298</td>
      <td>0.398</td><td>0.239</td><td>0.645</td></tr>
    <tr><td>Gen. attack (our)</td><td>PatchTST</td>
      <td>0.822</td><td>0.8</td><td>0.146</td>
      <td>0.086</td><td>0.152</td><td>0.814</td></tr>

    <!-- ResCNN -->
    <tr>
      <td rowspan="9">ResCNN</td>
      <td>Unattacked</td><td>-</td>
      <td>0</td><td>0.079</td><td>0.93</td>
      <td>0</td><td>0.003</td><td>0.995</td>
    </tr>
    <tr><td>FGSM</td><td>-</td>
      <td>0.448</td><td>0.487</td><td>0.524</td>
      <td>0.385</td><td>0.267</td><td>0.617</td></tr>
    <tr><td>iFGSM</td><td>-</td>
      <td>0.842</td><td><b><u>0.921</u></b></td><td><b><u>0.081</u></b></td>
      <td><b><u>0.918</u></b></td><td><b><u>0.843</u></b></td><td><b><u>0.089</u></b></td></tr>
    <tr><td>Iterative Gen. Attack (our)</td><td>LSTM</td>
      <td>0.103</td><td>0.114</td><td>0.913</td>
      <td>0.296</td><td>0.265</td><td>0.632</td></tr>
    <tr><td>Iterative Gen. Attack (our)</td><td>ResCNN</td>
      <td>0.642</td><td>0.567</td><td>0.37</td>
      <td>0.342</td><td>0.275</td><td>0.591</td></tr>
    <tr><td>Iterative Gen. Attack (our)</td><td>PatchTST</td>
      <td><u>0.857</u></td><td><u>0.869</u></td><td><u>0.091</u></td>
      <td><u>0.653</u></td><td><u>0.567</u></td><td><u>0.314</u></td></tr>
    <tr><td>Gen. attack (our)</td><td>LSTM</td>
      <td>0.004</td><td>0.071</td><td>0.922</td>
      <td>0.002</td><td>0.002</td><td>0.995</td></tr>
    <tr><td>Gen. attack (our)</td><td>ResCNN</td>
      <td><b><u>0.909</u></b></td><td>0.837</td><td>0.148</td>
      <td>0.463</td><td>0.331</td><td>0.538</td></tr>
    <tr><td>Gen. attack (our)</td><td>PatchTST</td>
      <td>0.766</td><td>0.779</td><td>0.183</td>
      <td>0.004</td><td>0.005</td><td>0.994</td></tr>

    <!-- PatchTST -->
    <tr>
      <td rowspan="9">PatchTST</td>
      <td>Unattacked</td><td>-</td>
      <td>0.067</td><td>0.083</td><td>0.906</td>
      <td>0</td><td>0.072</td><td>0.908</td>
    </tr>
    <tr><td>FGSM</td><td>-</td>
      <td>0.65</td><td>0.512</td><td>0.294</td>
      <td>0.359</td><td>0.316</td><td>0.552</td></tr>
    <tr><td>iFGSM</td><td>-</td>
      <td><b><u>0.911</u></b></td><td><b><u>1</u></b></td><td><b><u>0</u></b></td>
      <td><b><u>0.668</u></b></td><td><u>0.409</u></td><td><b><u>0.246</u></b></td></tr>
    <tr><td>Iterative Gen. Attack (our)</td><td>LSTM</td>
      <td>0.367</td><td>0.317</td><td>0.694</td>
      <td>0.519</td><td>0.221</td><td>0.52</td></tr>
    <tr><td>Iterative Gen. Attack (our)</td><td>ResCNN</td>
      <td><u>0.767</u></td><td><u>0.712</u></td><td><u>0.217</u></td>
      <td><u>0.591</u></td><td><b><u>0.426</u></b></td><td>0.396</td></tr>
    <tr><td>Iterative Gen. Attack (our)</td><td>PatchTST</td>
      <td>0.406</td><td>0.324</td><td>0.689</td>
      <td>0.42</td><td>0.224</td><td>0.558</td></tr>
    <tr><td>Gen. attack (our)</td><td>LSTM</td>
      <td>0.067</td><td>0.084</td><td>0.917</td>
      <td>0.231</td><td>0.258</td><td>0.601</td></tr>
    <tr><td>Gen. attack (our)</td><td>ResCNN</td>
      <td>0.756</td><td>0.679</td><td>0.222</td>
      <td>0.581</td><td>0.38</td><td><u>0.272</u></td></tr>
    <tr><td>Gen. attack (our)</td><td>PatchTST</td>
      <td>0</td><td>0.061</td><td>0.939</td>
      <td>0</td><td>0.181</td><td>0.911</td></tr>
  </tbody>
</table>
</div>

Results after attacks for different target models and datasets. The best result for a dataset and target model is highlighted in bold, while the second best result is underlined.
