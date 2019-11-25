# ACClip-Pytorch
The PyTorch implementation of ICLR2020 submission: Why ADAM Beats SGD for Attention Models (https://openreview.net/pdf?id=SJx37TEtDH)


## Requirments
Strongly recommend that you have a Anaconda enviroment. (https://www.anaconda.com/distribution/)
* Python >= 3.6
* PyTorch >= 1.0  (https://pytorch.org/get-started/locally/)
* torchtext >= 0.4.0 (conda install -c pytorch torchtext)  
* numpy >= 1.16.4
* matplotlib >= 3.1.0
* transformers >= 2.1.1 (pip install transformers)

## Setup

### Text Classification
```shell script
$ python run_text_classifier.py --optimizer=acclip --lr=0.01 --epoch=20
```
optmizers chosen from "acclip; adam; sgd"  

### GLUE task
* Donwload the datasets
```shell script
$ python download_mrpc_data.py
```

* Run Bert model, where you can specify your optimizers in ```run_mrpc.sh```  
```shell script
$ sh ./run_mrpc.sh
```

## Plot for visualization
For training, the evaluation results will be written to the `curves` folder. 
Install Jupyter notebook to run ```plot.ipynb```.
