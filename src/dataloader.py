import re
import os
import sys
import random
import string
import logging
import argparse
from shutil import copyfile
from datetime import datetime
from collections import Counter
import torch
import msgpack


def load_data(opt):
    with open('SQuAD/meta.msgpack', 'rb') as f:
        meta = msgpack.load(f, encoding='utf8')

    if opt['glove_char_embedding'] == True:
        embedding = torch.Tensor(meta['glove_char_embedding'])
    else:
        embedding = torch.Tensor(meta['embedding'])

    #char_embedding = torch.Tensor(meta['char_embeddings'])
    opt['pretrained_words'] = True
    opt['vocab_size'] = embedding.size(0)
    opt['embedding_dim'] = embedding.size(1)
    opt['pos_size'] = len(meta['vocab_tag'])
    opt['ner_size'] = len(meta['vocab_ent'])
    with open(opt['data_file'], 'rb') as f:
        data = msgpack.load(f, encoding='utf8')
    train = data['train']
    data['dev'].sort(key=lambda x: len(x[1]))
    dev = [x[:-1] for x in data['dev']]
    dev_y = [x[-1] for x in data['dev']]
    return train, dev, dev_y, embedding, opt#, char_embedding

class BatchGen:
    def __init__(self, opt, data, batch_size, gpu, evaluation=False):
        """
        input:
            data(train/dev) - list of lists
            batch_size - int
        # train: id, context_id, context_features, tag_id, ent_id, iob_np_ids, iob_ner_ids,
        #        question_id, context, context_token_span, answer_start, answer_end
        """
        self.opt = opt
        self.batch_size = batch_size
        self.eval = evaluation
        self.gpu = gpu

        # shuffle
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        # chunk into batches
        data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        self.data = data

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for batch in self.data:
            batch_size = len(batch)
            batch = list(zip(*batch))
            if self.eval:
                assert len(batch) == 10
            else:
                assert len(batch) == 12 # ## TODO: change here if add more features

            context_len = max(len(x) for x in batch[1])
            context_id = torch.LongTensor(batch_size, context_len).fill_(0)
            for i, doc in enumerate(batch[1]):
                context_id[i, :len(doc)] = torch.LongTensor(doc)

            feature_len = len(batch[2][0][0])

            context_feature = torch.Tensor(batch_size, context_len, feature_len).fill_(0)
            for i, doc in enumerate(batch[2]):
                for j, feature in enumerate(doc):
                    context_feature[i, j, :] = torch.Tensor(feature)

            context_tag = torch.Tensor(batch_size, context_len, self.opt['pos_size']).fill_(0)
            for i, doc in enumerate(batch[3]):
                for j, tag in enumerate(doc):
                    context_tag[i, j, tag] = 1

            context_ent = torch.Tensor(batch_size, context_len, self.opt['ner_size']).fill_(0)
            for i, doc in enumerate(batch[4]):
                for j, ent in enumerate(doc):
                    context_ent[i, j, ent] = 1

            ### add new feature here ###
            context_iob_np = torch.Tensor(batch_size, context_len, self.opt['iob_np_size']).fill_(0)
            for i, doc in enumerate(batch[5]):
                for j, iob_np in enumerate(doc):
                    context_iob_np[i, j, iob_np] = 1

            context_iob_ner = torch.Tensor(batch_size, context_len, self.opt['iob_ner_size']).fill_(0)
            for i, doc in enumerate(batch[6]):
                for j, iob_ner in enumerate(doc):
                    context_iob_ner[i, j, iob_ner] = 1


            question_len = max(len(x) for x in batch[7])
            question_id = torch.LongTensor(batch_size, question_len).fill_(0)
            for i, doc in enumerate(batch[7]):
                question_id[i, :len(doc)] = torch.LongTensor(doc)

            # mask: if id is 0, then mask is 1, otherwise mask is 0
            # in question_id and context_id, the 0 means padding
            context_mask = torch.eq(context_id, 0)
            question_mask = torch.eq(question_id, 0)
            text = list(batch[8])
            span = list(batch[9])
            if not self.eval:
                y_s = torch.LongTensor(batch[10])
                y_e = torch.LongTensor(batch[11])
            if self.gpu:
                context_id = context_id.pin_memory()
                context_feature = context_feature.pin_memory()
                context_tag = context_tag.pin_memory()
                context_ent = context_ent.pin_memory()
                context_iob_np = context_iob_np.pin_memory()
                context_iob_ner = context_iob_ner.pin_memory()
                context_mask = context_mask.pin_memory()
                question_id = question_id.pin_memory()
                question_mask = question_mask.pin_memory()
            if self.eval:
                yield (context_id, context_feature, context_tag, context_ent, context_iob_np, context_iob_ner, context_mask,
                       question_id, question_mask, text, span)
            else:
                yield (context_id, context_feature, context_tag, context_ent, context_iob_np, context_iob_ner,context_mask,
                       question_id, question_mask, y_s, y_e, text, span)
