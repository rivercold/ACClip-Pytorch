# ACClip-Pytorch
The PyTorch implementation of ICLR2020 submission: Why ADAM Beats SGD for Attention Models (https://openreview.net/pdf?id=SJx37TEtDH)


## Requirments
* Python >= 3.6
* PyTorch >= 1.0
* torchtext >= 0.4.0
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
$ python download_glue_data.py
```

* Run Bert model, where you can specify your optimizers in ```run_glue.sh```  
```shell script
$ sh ./run_glue.sh
```

## Plot for visualization

Install Jupyter notebook to run ```plot.ipynb```.

## TODO List

Check here for implementing ACClip for Transformers: https://github.com/huggingface/transformers/blob/master/transformers/optimization.py
