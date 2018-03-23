import re
import os
import json
import spacy
import msgpack
import unicodedata
import numpy as np
import argparse
import collections
import multiprocessing
from tqdm import tqdm
import logging
from utils import str2bool
import torch
from torch.autograd import Variable
import sys
import random
import string
import argparse
from shutil import copyfile
from datetime import datetime
from collections import Counter
from dataloader import *



parser = argparse.ArgumentParser(
    description='Train a Document Reader model.'
)
# system
parser.add_argument('--log_file', default='output.log',
                    help='path for log file.')
parser.add_argument('--log_per_updates', type=int, default=3,
                    help='log model loss per x updates (mini-batches).')
parser.add_argument('--data_file', default='SQuAD/data.msgpack',
                    help='path to preprocessed data file.')
parser.add_argument('--model_dir', default='models',
                    help='path to store saved models.')
parser.add_argument('--save_last_only', action='store_true',
                    help='only save the final models.')
parser.add_argument('--eval_per_epoch', type=int, default=1,
                    help='perform evaluation per x epochs.')
parser.add_argument('--seed', type=int, default=1013,
                    help='random seed for data shuffling, dropout, etc.')
parser.add_argument("--cuda", type=str2bool, nargs='?',
                    const=True, default=torch.cuda.is_available(),
                    help='whether to use GPU acceleration.')
# training
parser.add_argument('-e', '--epochs', type=int, default=40)
parser.add_argument('-bs', '--batch_size', type=int, default=32)
parser.add_argument('-rs', '--resume', default='',
                    help='previous model file name (in `model_dir`). '
                         'e.g. "checkpoint_epoch_11.pt"')
parser.add_argument('-ro', '--resume_options', action='store_true',
                    help='use previous model options, ignore the cli and defaults.')
parser.add_argument('-rlr', '--reduce_lr', type=float, default=0.,
                    help='reduce initial (resumed) learning rate by this factor.')
parser.add_argument('-op', '--optimizer', default='adamax',
                    help='supported optimizer: adamax, sgd')
parser.add_argument('-gc', '--grad_clipping', type=float, default=10)
parser.add_argument('-wd', '--weight_decay', type=float, default=0)
parser.add_argument('-lr', '--learning_rate', type=float, default=0.1,
                    help='only applied to SGD.')
parser.add_argument('-mm', '--momentum', type=float, default=0,
                    help='only applied to SGD.')
parser.add_argument('-tp', '--tune_partial', type=int, default=1000,
                    help='finetune top-x embeddings.')
parser.add_argument('--fix_embeddings', action='store_true',
                    help='if true, `tune_partial` will be ignored.')
parser.add_argument('--rnn_padding', action='store_true',
                    help='perform rnn padding (much slower but more accurate).')
# model
parser.add_argument('--question_merge', default='self_attn')
parser.add_argument('--doc_layers', type=int, default=3)
parser.add_argument('--question_layers', type=int, default=3)
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--num_features', type=int, default=4)
parser.add_argument('--pos', type=str2bool, nargs='?', const=True, default=True,
                    help='use pos tags as a feature.')
parser.add_argument('--ner', type=str2bool, nargs='?', const=True, default=True,
                    help='use named entity tags as a feature.')
parser.add_argument('--use_qemb', type=str2bool, nargs='?', const=True, default=True)
parser.add_argument('--concat_rnn_layers', type=str2bool, nargs='?',
                    const=True, default=True)
parser.add_argument('--dropout_emb', type=float, default=0.4)
parser.add_argument('--dropout_rnn', type=float, default=0.4)
parser.add_argument('--dropout_rnn_output', type=str2bool, nargs='?',
                    const=True, default=True)
parser.add_argument('--max_len', type=int, default=15)
parser.add_argument('--rnn_type', default='lstm',
                    help='supported types: rnn, gru, lstm')

args = parser.parse_args()

# set model dir
model_dir = args.model_dir
os.makedirs(model_dir, exist_ok=True)
model_dir = os.path.abspath(model_dir)

# set random seed
random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# setup logger
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
fh = logging.FileHandler(args.log_file)
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
log.addHandler(fh)
log.addHandler(ch)




def main():
    log.info('[program starts.]')
    train, dev, dev_y, embedding, opt = load_data(vars(args))
    log.info(opt)
    log.info('[Data loaded.]')

    # test dataloader 
    batches = BatchGen(opt, train, batch_size=1, gpu=False)

    for i, batch in enumerate(batches):
        inputs = [Variable(e) for e in batch[:7]]

    #question_mask
    print('question_mask of last input in the batch is {}'.format(inputs[6]))

    # text
    print('text of last input in the batch is {}'.format(batch[9]))





if __name__ == '__main__':
    main()


