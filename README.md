# ACClip-Pytorch
The PyTorch implementation of ICLR2020 submission: Why ADAM Beats SGD for Attention Models (https://openreview.net/pdf?id=SJx37TEtDH)  

Github Repo: https://github.com/rivercold/ACClip-Pytorch  

## Requirments
Strongly recommend that you have a Anaconda enviroment. (https://www.anaconda.com/distribution/)
* Python >= 3.6
* PyTorch >= 1.0  (https://pytorch.org/get-started/locally/)
* torchtext >= 0.4.0 (conda install -c pytorch torchtext)  
* numpy >= 1.16.4
* matplotlib >= 3.1.0
* transformers >= 2.1.1 (pip install transformers)

## Core Implementaions
`ACClip` optimizer:  https://github.com/rivercold/ACClip-Pytorch/blob/master/optimizers/ACClip.py  
`print_noise` function for LSTM: https://github.com/rivercold/ACClip-Pytorch/blob/master/run_text_classifier.py#L22
`print_noise` function for BERT: https://github.com/rivercold/ACClip-Pytorch/blob/master/run_mrpc.py#L224

## Setup

### Text Classification

#### IMDB
```shell script
$ python run_text_classifier.py --optimizer=acclip --lr=0.001 --epoch=30
```
optmizers chosen from `acclip`, `adam` and `sgd`.  

##### Plot Noise 
```shell script
$ python run_text_classifier.py --optimizer=sgd --lr=0.1 --epoch=30 --mode=plot
```

#### SST
```shell script
$ python run_text_classifier.py --optimizer=acclip --lr=0.001 --epoch=20 --dataset=sst
```

### GLUE task
* Donwload the datasets
```shell script
$ python download_mrpc_data.py
```

* Run Bert model, where you can specify your optimizers and learning rates in ```run_mrpc.sh```  
```shell script
$ sh ./run_mrpc.sh
```

* To plot noise
switch `do_train` with `do_plot` in `./run_mrpc.sh` 

## Plot for visualization
For training, the evaluation results will be written to the `curves` folder.   
Install Jupyter notebook to run ```plot.ipynb```.  

For plotting noise norm, the evaluation results will be written to `noises` folder.   
Run ```noise_plot.ipynb```
