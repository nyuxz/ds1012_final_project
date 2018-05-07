import torch
import torch.nn as nn
from . import layers

# Modification: add 'pos' and 'ner' features.
# Origin: https://github.com/facebookresearch/ParlAI/tree/master/parlai/agents/drqa


class RnnDocReader(nn.Module):
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

    def __init__(self, opt, padding_idx=0, embedding=None):
        super(RnnDocReader, self).__init__()
        # Store config
        self.opt = opt

        # Word embeddings
        if opt['pretrained_words']:
            assert embedding is not None
            self.embedding = nn.Embedding(embedding.size(0),
                                          embedding.size(1),
                                          padding_idx=padding_idx)
            self.embedding.weight.data[2:, :] = embedding[2:, :]
            if opt['fix_embeddings']:
                assert opt['tune_partial'] == 0
                for p in self.embedding.parameters():
                    p.requires_grad = False
            elif opt['tune_partial'] > 0: # fine-tune 1000 most frequent [question] words
                assert opt['tune_partial'] + 2 < embedding.size(0)
                fixed_embedding = embedding[opt['tune_partial'] + 2:] # other words embedding keep fixed
                self.register_buffer('fixed_embedding', fixed_embedding)
                self.fixed_embedding = fixed_embedding
        else:  # random initialized
            self.embedding = nn.Embedding(opt['vocab_size'],
                                          opt['embedding_dim'],
                                          padding_idx=padding_idx)
        # Projection for attention weighted question
        if opt['use_qemb']:
            self.qemb_match = layers.SeqAttnMatch(opt['embedding_dim'])

        # Input size to RNN: word emb + question emb + manual features + new features
        doc_input_size = opt['embedding_dim'] + opt['num_features']
        if opt['use_qemb']:
            doc_input_size += opt['embedding_dim']
        if opt['pos']:
            doc_input_size += opt['pos_size']
        if opt['ner']:
            doc_input_size += opt['ner_size']
        # new embedding layers /token features
        if opt['iob_np']:
            doc_input_size += opt['iob_np_size'] #3
        if opt['iob_ner']:
            doc_input_size += opt['iob_ner_size'] #3
        if opt['wwwwh']:
            doc_input_size += opt['wwwwh_size'] # 6
        if opt['part_ner']:
            doc_input_size += opt['part_ner_size'] #2


        # question size to RNN: word emb + question emb + manual features + new features
        question_size = opt['embedding_dim']
        if opt['pos']:
            question_size += opt['q_pos_size']
        if opt['ner']:
            question_size += opt['q_ner_size']
        if opt['iob_np']:
            question_size += opt['q_iob_np_size']
        if opt['iob_ner']:
            question_size += opt['q_iob_ner_size']

        # RNN document encoder
        self.doc_rnn = layers.StackedBRNN(
            input_size=doc_input_size,
            hidden_size=opt['hidden_size'],
            num_layers=opt['doc_layers'],
            dropout_rate=opt['dropout_rnn'],
            dropout_output=opt['dropout_rnn_output'],
            concat_layers=opt['concat_rnn_layers'],
            rnn_type=self.RNN_TYPES[opt['rnn_type']],
            padding=opt['rnn_padding'],
        )

        # RNN question encoder

        if opt['multi_level_question']:
            # RNN question encoder(adding new feature to question also)
            self.question_rnn = layers.StackedBRNN(
                input_size=question_size, #here need change
                hidden_size=opt['hidden_size'],
                num_layers=opt['question_layers'],
                dropout_rate=opt['dropout_rnn'],
                dropout_output=opt['dropout_rnn_output'],
                concat_layers=opt['concat_rnn_layers'],
                rnn_type=self.RNN_TYPES[opt['rnn_type']],
                padding=opt['rnn_padding'],
            )
        else:
            self.question_rnn = layers.StackedBRNN(
                input_size=opt['embedding_dim'],
                hidden_size=opt['hidden_size'],
                num_layers=opt['question_layers'],
                dropout_rate=opt['dropout_rnn'],
                dropout_output=opt['dropout_rnn_output'],
                concat_layers=opt['concat_rnn_layers'],
                rnn_type=self.RNN_TYPES[opt['rnn_type']],
                padding=opt['rnn_padding'],
            )


        # Output sizes of rnn encoders
        doc_hidden_size = 2 * opt['hidden_size']
        question_hidden_size = 2 * opt['hidden_size']
        if opt['concat_rnn_layers']:
            doc_hidden_size *= opt['doc_layers']
            question_hidden_size *= opt['question_layers']

        # Question merging
        if opt['question_merge'] not in ['avg', 'self_attn']:
            raise NotImplementedError('question_merge = %s' % opt['question_merge'])
        if opt['question_merge'] == 'self_attn':
            self.self_attn = layers.LinearSeqAttn(question_hidden_size)

        # Bilinear attention for span start/end
        self.start_attn = layers.BilinearSeqAttn(
            doc_hidden_size,
            question_hidden_size,
        )
        self.end_attn = layers.BilinearSeqAttn(
            doc_hidden_size,
            question_hidden_size,
        )


    def forward(self, x1, x1_f, x1_pos, x1_ner, x1_iob_np, x1_iob_ner,x1_part_ner,
                x2_pos, x2_ner, x2_iob_np, x2_iob_ner, x1_mask, x2,x2_mask):
        """Inputs:
        context_id, context_feature, context_tag, context_ent, context_iob_np, context_iob_ner, context_part_ner,
        question_tag,question_ent,question_iob_np,question_iob_ner, context_mask, question_id, question_mask,
        x1 = document word indices             [batch * len_d]
        x1_f = document word features indices  [batch * len_d * nfeat]
        x1_pos = document POS tags             [batch * len_d]
        x1_ner = document entity tags          [batch * len_d]
        x1_iob_np
        x1_iob_ner
        x1_part_ner
        x2_pos
        x2_ner
        x2_iob_np
        x2_iob_ner
        x1_mask = document padding mask        [batch * len_d]
        x2 = question word indices             [batch * len_q]
        x2_mask = question padding mask        [batch * len_q]
        """

        if len(x2_pos) == 0: # no embedding for question
            # Embed both document and question
            x1_emb = self.embedding(x1)
            x2_emb = self.embedding(x2)

            # Dropout on embeddings
            if self.opt['dropout_emb'] > 0:
                x1_emb = nn.functional.dropout(x1_emb, p=self.opt['dropout_emb'],
                                               training=self.training)
                x2_emb = nn.functional.dropout(x2_emb, p=self.opt['dropout_emb'],
                                               training=self.training)

            #### multi-level embedding for context
            drnn_input_list = [x1_emb, x1_f]
            # Add attention-weighted question representation
            if self.opt['use_qemb']:
                x2_weighted_emb = self.qemb_match(x1_emb, x2_emb, x2_mask)
                drnn_input_list.append(x2_weighted_emb)
            if self.opt['pos']:
                drnn_input_list.append(x1_pos)
            if self.opt['ner']:
                drnn_input_list.append(x1_ner)
            if self.opt['iob_np']:
                drnn_input_list.append(x1_iob_np)
            if self.opt['iob_ner']:
                drnn_input_list.append(x1_iob_ner)
            if self.opt['part_ner']:
                drnn_input_list.append(x1_part_ner)
            drnn_input = torch.cat(drnn_input_list, 2)

            # Encode document with RNN
            doc_hiddens = self.doc_rnn(drnn_input, x1_mask)

            # Encode question with RNN + merge hiddens
            question_hiddens = self.question_rnn(x2_emb, x2_mask)
            if self.opt['question_merge'] == 'avg':
                q_merge_weights = layers.uniform_weights(question_hiddens, x2_mask)
            elif self.opt['question_merge'] == 'self_attn':
                q_merge_weights = self.self_attn(question_hiddens, x2_mask)
            question_hidden = layers.weighted_avg(question_hiddens, q_merge_weights)

            # Predict start and end positions
            start_scores = self.start_attn(doc_hiddens, question_hidden, x1_mask)
            end_scores = self.end_attn(doc_hiddens, question_hidden, x1_mask)
            return start_scores, end_scores

        else: #embedding for question
            # Embed both document and question
            x1_emb = self.embedding(x1)
            x2_emb = self.embedding(x2)

            # Dropout on embeddings
            if self.opt['dropout_emb'] > 0:
                x1_emb = nn.functional.dropout(x1_emb, p=self.opt['dropout_emb'],
                                               training=self.training)
                x2_emb = nn.functional.dropout(x2_emb, p=self.opt['dropout_emb'],
                                               training=self.training)

            #### multi-level embedding for context
            drnn_input_list = [x1_emb, x1_f]
            # Add attention-weighted question representation
            if self.opt['use_qemb']:
                x2_weighted_emb = self.qemb_match(x1_emb, x2_emb, x2_mask)
                drnn_input_list.append(x2_weighted_emb)
            if self.opt['pos']:
                drnn_input_list.append(x1_pos)
            if self.opt['ner']:
                drnn_input_list.append(x1_ner)
            if self.opt['iob_np']:
                drnn_input_list.append(x1_iob_np)
            if self.opt['iob_ner']:
                drnn_input_list.append(x1_iob_ner)
            if self.opt['part_ner']:
                drnn_input_list.append(x1_part_ner)
            drnn_input = torch.cat(drnn_input_list, 2)

            # Encode document with RNN, here we feed multi-level embedding to RNN, encode as hidden cell
            doc_hiddens = self.doc_rnn(drnn_input, x1_mask)

            ### multi-level embedding for question
            drnn_question_list = [x2_emb]
            if self.opt['pos']:
                drnn_question_list.append(x2_pos)
            if self.opt['ner']:
                drnn_question_list.append(x2_ner)
            if self.opt['iob_np']:
                drnn_question_list.append(x2_iob_np)
            if self.opt['iob_ner']:
                drnn_question_list.append(x2_iob_ner)

            drnn_question = torch.cat(drnn_question_list, 2)

            # Encode question with RNN + merge hiddens
            question_hiddens = self.question_rnn(drnn_question, x2_mask)
            if self.opt['question_merge'] == 'avg':
                q_merge_weights = layers.uniform_weights(question_hiddens, x2_mask)
            elif self.opt['question_merge'] == 'self_attn':
                q_merge_weights = self.self_attn(question_hiddens, x2_mask)
            question_hidden = layers.weighted_avg(question_hiddens, q_merge_weights)

            # Predict start and end positions
            start_scores = self.start_attn(doc_hiddens, question_hidden, x1_mask)
            end_scores = self.end_attn(doc_hiddens, question_hidden, x1_mask)
            return start_scores, end_scores
