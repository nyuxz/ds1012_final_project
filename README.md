Multi-level Embedding Representation for Reading Comprehension
---
[Paper]: A pytorch implementation of the ACL 2017 paper [Reading Wikipedia to Answer Open-Domain Questions](http://www-cs.stanford.edu/people/danqi/papers/acl2017.pdf) (DrQA).     

[Code]: Our code id adopt from https://github.com/hitvoice/DrQA     

## Data
We conduct our experiments on SQuAD. [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) is a reading comprehension benchmark.     

## Model Architecture     



## Requirements
- python 3.5 
- pytorch 0.3
- numpy
- msgpack
- spacy 2.0

## Set up     
to download data and Glove     
```python
bash download.sh
```     

to download Pre-trained character-level vector     
http://www.logos.t.u-tokyo.ac.jp/~hassy/publications/arxiv2016jmt/jmt_pre-trained_embeddings.tar.gz


### Train

```bash
# prepare the data
python src/prepro.py
# train for 40 epochs with batchsize 32
python src/main.py 
```
