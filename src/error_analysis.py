import re
import os
import json
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
import spacy
from evaluation import *
#from prepro import flatten_json


parser = argparse.ArgumentParser(
    description='error analysis'
)
parser.add_argument("--cuda", type=str2bool, nargs='?',
                    const=True, default=torch.cuda.is_available(),
                    help='whether to use GPU acceleration.')
parser.add_argument('--testmodel', default='models/all_with_q/best_model.pt',
                    help='path to model file')
parser.add_argument('--save_result', default='all_with_q',
                    help='path to save file')

args = parser.parse_args()


def flatten_json(data_file, mode):
    """Flatten each article in training data."""
    with open(data_file) as f:
        data = json.load(f)['data']
    rows = []
    for article in data:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                id_, question, answers = qa['id'], qa['question'], qa['answers']
                if mode == 'train':
                    answer = answers[0]['text']  # in training data there's only one answer
                    answer_start = answers[0]['answer_start'] # char level length
                    answer_end = answer_start + len(answer) # char level lenght
                    rows.append((id_, context, question, answer, answer_start, answer_end))
                else:  # mode == 'dev'
                    answers = [a['text'] for a in answers]
                    rows.append((id_, context, question, answers))
    return rows


def main():

    if args.cuda:
        checkpoint = torch.load(args.testmodel)
    else:
        checkpoint = torch.load(args.testmodel, map_location=lambda storage, loc: storage)

    state_dict = checkpoint['state_dict']
    opt = checkpoint['config']
    dev_file = 'SQuAD/dev-v1.1.json'
    dev = flatten_json(dev_file, 'dev')

    with open('SQuAD/meta.msgpack', 'rb') as f:
        meta = msgpack.load(f, encoding='utf8')

    with open(opt['data_file'], 'rb') as f:
        data = msgpack.load(f, encoding='utf8')

    embedding = torch.Tensor(meta['glove_char_embedding'])
    opt['cuda'] = args.cuda

    dev = [x[:-1] for x in data['dev']]
    dev_y = [x[-1] for x in data['dev']]


    model = DocReaderModel(opt, embedding, state_dict)
    if args.cuda:
        model.cuda()

    result_list = []
    number_error = 0
    error_id_list = []
    id = 0
    number_cases = len(dev)
    batches = iter(BatchGen(opt, dev, batch_size=1, gpu=args.cuda, evaluation=True))
    for id in range(number_cases):
        input = next(batches)
        predictions = model.predict(input)
        predict_answer = predictions[0]
        answer = dev_y[id]
        em_score = _exact_match(predict_answer, answer)
        f1_score = _f1_score(predict_answer, answer)
        print('------ Case: {} ------:'.format(id))
        print('Predictions list:{}'.format(predictions[0:5]))
        print('Predict Answer: {}'.format(predict_answer))
        print('True Answer: {}'.format(answer))
        print('examt match: {}'.format(em_score))
        print('f1 score: {}'.format(f1_score))

        result = {}
        result["case_id"]= id
        result["predict_answer"]= predict_answer
        result["true_answer"]= answer
        result["em_score"]= em_score
        result["f1_score"]= f1_score
        result_list.append(result)

        if f1_score < 0.5:
            number_error +=1
            error_id_list.append(id)
        id += 1
    print(opt)
    return error_id_list, number_error/number_cases, result_list

if __name__ == '__main__':
    #test_loader()
    error_id_list, error_percentage, result_list = main()

    with open('results/' + args.save_result +'_result.txt', 'w') as outfile:
        json.dump(result_list, outfile)

    #print('case id with error answer: {}'.format(error_id_list))
    print('error percentage: {}'.format(error_percentage))
