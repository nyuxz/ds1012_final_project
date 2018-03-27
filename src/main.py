import re
import os
import json
#import spacy
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
from drqa.model import DocReaderModel
from evaluation import *



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


def test_loader():
    '''
    this function used to test dataloader and print input data/check dimensions 
    '''

    log.info('[program starts.]')
    train, dev, dev_y, embedding, opt = load_data(vars(args))
    log.info(opt)
    log.info('[Data loaded.]')

    # test dataloader 
    batches = BatchGen(opt, train, batch_size=1, gpu=False)

    for i, batch in enumerate(batches):
        inputs = [Variable(e) for e in batch[:7]]

    print('context_id{}'.format(inputs[0]))
    print('.............................................................')

    print('context_feature{}'.format(inputs[1]))
    print('.............................................................')

    print('context_tag POS{}'.format(inputs[2]))
    print('.............................................................')

    print('context_ent{}'.format(inputs[3]))
    print('.............................................................')

    print('context_mask{}'.format(inputs[4]))
    print('.............................................................')

    print('qquestion_id{}'.format(inputs[5]))
    print('.............................................................')

    print('question_mask{}'.format(inputs[6]))
    print('.............................................................')

    #print('y_s{}'.format(inputs[7]))
    #print('.............................................................')

    #print('y_e{}'.format(inputs[8]))
    #print('.............................................................')

    print('text{}'.format(batch[9]))
    print('.............................................................')

    print('span{}'.format(batch[10]))
    print('.............................................................')


def main():
    log.info('[program starts.]')
    train, dev, dev_y, embedding, opt = load_data(vars(args))
    log.info(opt)
    log.info('[Data loaded.]')

    if args.resume:
        log.info('[loading previous model...]')
        checkpoint = torch.load(os.path.join(model_dir, args.resume))
        if args.resume_options:
            opt = checkpoint['config']
        state_dict = checkpoint['state_dict']
        model = DocReaderModel(opt, embedding, state_dict)
        epoch_0 = checkpoint['epoch'] + 1
        for i in range(checkpoint['epoch']):
            # synchronize random seed
            random.setstate(checkpoint['random_state'])
            torch.random.set_rng_state(checkpoint['torch_state'])
            torch.cuda.set_rng_state(checkpoint['torch_cuda_state'])
        if args.reduce_lr:
            lr_decay(model.optimizer, lr_decay=args.reduce_lr)
    else:
        model = DocReaderModel(opt, embedding)
        epoch_0 = 1

    if args.cuda:
        model.cuda()

    if args.resume:
        batches = BatchGen(opt, dev, batch_size=args.batch_size, evaluation=True, gpu=args.cuda)
        predictions = []
        for batch in batches:
            predictions.extend(model.predict(batch))
        em, f1 = score(predictions, dev_y)
        log.info("[dev EM: {} F1: {}]".format(em, f1))
        best_val_score = f1
    else:
        best_val_score = 0.0

    for epoch in range(epoch_0, epoch_0 + args.epochs):
        log.warning('Epoch {}'.format(epoch))
        # train
        batches = BatchGen(opt, train, batch_size=args.batch_size, gpu=args.cuda)
        start = datetime.now()
        for i, batch in enumerate(batches):
            model.update(batch)
            if i % args.log_per_updates == 0:
                log.info('epoch [{0:2}] updates[{1:6}] train loss[{2:.5f}] remaining[{3}]'.format(
                    epoch, model.updates, model.train_loss.value,
                    str((datetime.now() - start) / (i + 1) * (len(batches) - i - 1)).split('.')[0]))
        # eval
        if epoch % args.eval_per_epoch == 0:
            batches = BatchGen(opt, dev, batch_size=args.batch_size, evaluation=True, gpu=args.cuda)
            predictions = []
            for batch in batches:
                predictions.extend(model.predict(batch))
            em, f1 = score(predictions, dev_y)
            log.warning("dev EM: {} F1: {}".format(em, f1))
        # save
        if not args.save_last_only or epoch == epoch_0 + args.epochs - 1:
            model_file = os.path.join(model_dir, 'checkpoint_epoch_{}.pt'.format(epoch))
            model.save(model_file, epoch)
            if f1 > best_val_score:
                best_val_score = f1
                copyfile(
                    model_file,
                    os.path.join(model_dir, 'best_model.pt'))
                log.info('[new best model saved.]')


if __name__ == '__main__':
    #test_loader()
    main()


