import random
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging

from torch.autograd import Variable
from utils import AverageMeter
from .rnn_reader import RnnDocReader #from drqa.rnn_reader import RnnDocReader #


# Modification:
#   - change the logger name
#   - save & load "state_dict"s of optimizer and loss meter
#   - save all random seeds
#   - change the dimension of inputs (for POS and NER features)
# Origin: https://github.com/facebookresearch/ParlAI/tree/master/parlai/agents/drqa

logger = logging.getLogger(__name__)


class DocReaderModel(object):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    def __init__(self, opt, embedding=None, state_dict=None):
        # Book-keeping.
        self.opt = opt
        self.updates = state_dict['updates'] if state_dict else 0
        self.train_loss = AverageMeter()
        if state_dict:
            self.train_loss.load(state_dict['loss'])

        # Building network.
        self.network = RnnDocReader(opt, embedding=embedding)
        if state_dict:
            new_state = set(self.network.state_dict().keys())
            for k in list(state_dict['network'].keys()):
                if k not in new_state:
                    del state_dict['network'][k]
            self.network.load_state_dict(state_dict['network'])

        # Building optimizer.
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if opt['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(parameters, opt['learning_rate'],
                                       momentum=opt['momentum'],
                                       weight_decay=opt['weight_decay'])
        elif opt['optimizer'] == 'adamax':
            self.optimizer = optim.Adamax(parameters,
                                          weight_decay=opt['weight_decay'])
        else:
            raise RuntimeError('Unsupported optimizer: %s' % opt['optimizer'])
        if state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer'])

    def update(self, ex):
        # Train mode
        self.network.train()
        '''
        ex: context_id, context_feature, context_tag, context_ent, context_iob_np,
            context_iob_ner, context_part_ner,
            context_mask, question_id, question_mask, y_s, y_e, text, span
        '''
        # Transfer to GPU
        if self.opt['cuda']:
            inputs = [Variable(e.cuda(async=True)) for e in ex[:10]] # [0, 1, 2, 3, 4, 5, 6, 7,8,9]
            target_s = Variable(ex[10].cuda(async=True))
            target_e = Variable(ex[11].cuda(async=True))
        else:
            inputs = [Variable(e) for e in ex[:10]]
            target_s = Variable(ex[10])
            target_e = Variable(ex[11])

        # Run forward
        score_s, score_e = self.network(*inputs)

        # Compute loss and accuracies
        loss = F.nll_loss(score_s, target_s) + F.nll_loss(score_e, target_e)
        self.train_loss.update(loss.data[0])

        # Clear gradients and run backward
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm(self.network.parameters(),
                                      self.opt['grad_clipping'])

        # Update parameters
        self.optimizer.step()
        self.updates += 1

        # Reset any partially fixed parameters (e.g. rare words)
        self.reset_parameters()

    def predict(self, ex):
        # Eval mode
        self.network.eval()

        # Transfer to GPU
        if self.opt['cuda']:
            inputs = [Variable(e.cuda(async=True), volatile=True)
                      for e in ex[:10]]
        else:
            inputs = [Variable(e, volatile=True) for e in ex[:10]]

        # Run forward
        score_s, score_e = self.network(*inputs)

        # Transfer to CPU/normal tensors for numpy ops
        score_s = score_s.data.cpu()
        score_e = score_e.data.cpu()

        # Get argmax text spans
        text = ex[-2]
        spans = ex[-1]
        predictions = []
        max_len = self.opt['max_len'] or score_s.size(1)
        for i in range(score_s.size(0)):
            scores = torch.ger(score_s[i], score_e[i])
            scores.triu_().tril_(max_len - 1)
            scores = scores.numpy()
            s_idx, e_idx = np.unravel_index(np.argmax(scores), scores.shape)
            s_offset, e_offset = spans[i][s_idx][0], spans[i][e_idx][1]
            predictions.append(text[i][s_offset:e_offset])

        return predictions

    def reset_parameters(self):
        # Reset fixed embeddings to original value
        if self.opt['tune_partial'] > 0:
            offset = self.opt['tune_partial'] + 2
            if offset < self.network.embedding.weight.data.size(0):
                self.network.embedding.weight.data[offset:] \
                    = self.network.fixed_embedding

    def save(self, filename, epoch):
        params = {
            'state_dict': {
                'network': self.network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'updates': self.updates,
                'loss': self.train_loss.state_dict()
            },
            'config': self.opt,
            'epoch': epoch,
            'random_state': random.getstate(),
            'torch_state': torch.random.get_rng_state(),
            'torch_cuda_state': torch.cuda.get_rng_state()
        }
        try:
            torch.save(params, filename)
            logger.info('model saved to {}'.format(filename))
        except BaseException:
            logger.warn('[ WARN: Saving failed... continuing anyway. ]')

    def cuda(self):
        self.network.cuda()
