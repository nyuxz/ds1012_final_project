import re
import json
import spacy
import msgpack
import unicodedata
import numpy as np
import argparse
import collections
import multiprocessing
from multiprocessing import Pool
from utils import str2bool
from tqdm import tqdm
import logging
from char_embedding import CharEmbedding

parser = argparse.ArgumentParser(
    description='Preprocessing data files, about 10 minitues to run.'
)
parser.add_argument('--wv_file', default='glove/glove.840B.300d.txt',
                    help='path to word vector file.')
parser.add_argument('--wv_dim', type=int, default=300,
                    help='word vector dimension.')
parser.add_argument('--wv_cased', type=str2bool, nargs='?',
                    const=True, default=True,
                    help='treat the words as cased or not.')
parser.add_argument('--sort_all', action='store_true',
                    help='sort the vocabulary by frequencies of all words. '
                         'Otherwise consider question words first.')
parser.add_argument('--sample_size', type=int, default=0,
                    help='size of sample data (for debugging).')
parser.add_argument('--threads', type=int, default=multiprocessing.cpu_count(),
                    help='number of threads for preprocessing.')
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size for multiprocess tokenizing and tagging.')

args = parser.parse_args()
trn_file = 'SQuAD/train-v1.1.json'
dev_file = 'SQuAD/dev-v1.1.json'
wv_file = args.wv_file #glove
wv_dim = args.wv_dim #default = 300


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG,
                    datefmt='%m/%d/%Y %I:%M:%S')
log = logging.getLogger(__name__)

log.info(vars(args))
log.info('start data preparing...')


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


train = flatten_json(trn_file, 'train')
dev = flatten_json(dev_file, 'dev')
log.info('json data flattened.')


def clean_spaces(text):
    """normalize spaces in a string."""
    text = re.sub(r'\s', ' ', text)
    return text


def normalize_text(text):
    return unicodedata.normalize('NFD', text)

nlp = None

def init():
    """initialize spacy in each process"""
    '''
    'en': Noun chunks are "base noun phrases" – flat phrases that have a noun as their head.
    parser=False or disable=['parser'] : don't need any of the syntactic information,
                                        and will make spaCy load and run much faster.
    '''
    global nlp
    nlp = spacy.load('en', parser=False)


#------------new functions for new features/embedding layers stuffs------------------
## iob tag for noun phrases
def iob_np_tag(tag_list):
    '''
    function for creating iob_np
    @in: a list of POS tags
    @out: iob_np tags
    '''
    iob_np = ['o_np'] * len(tag_list)
    for i in range(len(tag_list)):
        if 'NN' in tag_list[i]:
            if iob_np[i-1] == 'b_np':
                iob_np[i] = 'i_np'
            elif iob_np[i-1] == 'i_np':
                iob_np[i] = 'i_np'
            else:
                iob_np[i] = 'b_np'
        i +=1
    return iob_np

## iob tag for NER
def iob_ner_tag(tag_list):
    '''
    function for creating iob_ner
    @in: a list of ner tags
    @out: iob_ner tags
    '''
    iob_ner = ['o_ner'] * len(tag_list)
    for i in range(len(tag_list)):
        if len(tag_list[i]) != 0:
            if iob_ner[i-1] == 'b_ner':
                iob_ner[i] = 'i_ner'
            elif iob_ner[i-1] == 'i_ner':
                iob_ner[i] = 'i_ner'
            else:
                iob_ner[i] = 'b_ner'
        i +=1
    return iob_ner

## Part of NER tag
stop_words = ['a', 'an', 'the', 'of', 'for', '\'s', 'For', 'The', 'A', 'An']
def part_ner_tag(tag_list, context_list):
    '''
    @in: a list of ner tags
    @out: part of ner tags
    '''
    ner_context = []
    part_ner = ['o_ner'] * len(tag_list)
    for i in range(len(tag_list)):
        if len(tag_list[i]) != 0 and context_list[i] not in stop_words:
            part_ner[i] = 'i_ner'
            ner_context.append(context_list[i])

    # combine lemma to ner_context list
    ner_context_str = ' '.join(ner_context)
    ner_context_ = nlp(ner_context_str)
    ner_context_lemma = [w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower() for w in ner_context_]
    ner_context_all = ner_context_lemma +  ner_context

    for j in range(len(context_list)):
        if context_list[j] in ner_context_all:
            part_ner[j] = 'i_ner'
    return part_ner
#--------------------------------------------------------------------------------

def annotate(row):
    '''
    notice: the tagging feature only apply on context
    '''
    global nlp
    id_, context, question = row[:3]
    q_doc = nlp(clean_spaces(question))
    c_doc = nlp(clean_spaces(context))
    question_tokens = [normalize_text(w.text) for w in q_doc]
    context_tokens = [normalize_text(w.text) for w in c_doc]
    question_tokens_lower = [w.lower() for w in question_tokens]
    context_tokens_lower = [w.lower() for w in context_tokens]
    context_token_span = [(w.idx, w.idx + len(w.text)) for w in c_doc] # the lenghth of each tokens
    #for context the new features
    context_tags = [w.tag_ for w in c_doc] # POS tagging
    context_ents = [w.ent_type_ for w in c_doc] # NER tagging
    context_iob_np = iob_np_tag(context_tags) # iob_np
    context_iob_ner = iob_ner_tag(context_ents) #iob_ner
    context_part_ner = part_ner_tag(context_ents, context_tokens) #part_ner

    #for question the new features
    question_tags = [w.tag_ for w in q_doc] # POS tagging
    question_ents = [w.ent_type_ for w in q_doc] # NER tagging
    question_iob_np = iob_np_tag(question_tags) # iob_np
    question_iob_ner = iob_ner_tag(question_ents) #iob_ner

    question_lemma = {w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower() for w in q_doc}
    # PRON is such as me/it/you
    # lemma_ : cats -> cat

    question_tokens_set = set(question_tokens)
    question_tokens_lower_set = set(question_tokens_lower)
    match_origin = [w in question_tokens_set for w in context_tokens]
    match_lower = [w in question_tokens_lower_set for w in context_tokens_lower]
    match_lemma = [(w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower()) in question_lemma for w in c_doc]
    # term frequency in document
    counter_ = collections.Counter(context_tokens_lower)
    total = len(context_tokens_lower)
    # frequent feature
    context_tf = [counter_[w] / total for w in context_tokens_lower]
    # exact match feature refering to the paper
    context_features = list(zip(match_origin, match_lower, match_lemma, context_tf))
    if not args.wv_cased:
        context_tokens = context_tokens_lower
        question_tokens = question_tokens_lower
    return (id_, context_tokens, context_features, context_tags, context_ents, context_iob_np, context_iob_ner, context_part_ner,
            question_tags,question_ents,question_iob_np,question_iob_ner,
            question_tokens, context, context_token_span) + row[3:]

################################
## TODO: add more fetures here #
################################


def index_answer(row):
    token_span = row[-4] #context_token_span
    starts, ends = zip(*token_span)
    answer_start = row[-2]
    answer_end = row[-1]
    try:
        return row[:-3] + (starts.index(answer_start), ends.index(answer_end))
    except ValueError:
        return row[:-3] + (None, None)

# adding the created features to the train/dev data
# with multiprocess(corresponding to number of cpu)
with Pool(args.threads, initializer=init) as p:
    train = list(tqdm(p.imap(annotate, train, chunksize=args.batch_size), total=len(train), desc='train'))
    dev = list(tqdm(p.imap(annotate, dev, chunksize=args.batch_size), total=len(dev), desc='dev  '))
train = list(map(index_answer, train))
initial_len = len(train)
train = list(filter(lambda x: x[-1] is not None, train))
log.info('drop {} inconsistent samples.'.format(initial_len - len(train)))
log.info('tokens generated')


# load vocabulary from word vector files (Glove)
wv_vocab = set()
with open(wv_file) as f:
    for line in f:
        token = normalize_text(line.rstrip().split(' ')[0])
        wv_vocab.add(token)
log.info('glove vocab loaded.')

def build_vocab(questions, contexts):
    """
    Build vocabulary sorted by global word frequency, or consider frequencies in questions first,
    which is controlled by `args.sort_all`.
    """
    if args.sort_all:
        counter = collections.Counter(w for doc in questions + contexts for w in doc)
        vocab = sorted([t for t in counter if t in wv_vocab], key=counter.get, reverse=True)
    else:
        counter_q = collections.Counter(w for doc in questions for w in doc)
        counter_c = collections.Counter(w for doc in contexts for w in doc)
        counter = counter_c + counter_q
        vocab = sorted([t for t in counter_q if t in wv_vocab], key=counter_q.get, reverse=True)
        vocab += sorted([t for t in counter_c.keys() - counter_q.keys() if t in wv_vocab],
                        key=counter.get, reverse=True)
    total = sum(counter.values())
    matched = sum(counter[t] for t in vocab)
    # out of vocb of pretrained glove
    log.info('vocab coverage {1}/{0} | OOV occurrence {2}/{3} ({4:.4f}%)'.format(
        len(counter), len(vocab), (total - matched), total, (total - matched) / total * 100))
    vocab.insert(0, "<PAD>") # in question_id and context_id, the 0 means padding
    vocab.insert(1, "<UNK>")
    return vocab, counter


full = train + dev
# row[5] = question_tokens, row[1] = context_tokens
vocab, counter = build_vocab([row[5] for row in full], [row[1] for row in full])

#pos
counter_tag = collections.Counter(w for row in full for w in row[3]) #context_tags
vocab_tag = sorted(counter_tag, key=counter_tag.get, reverse=True) # high rank with larger count

#ner
counter_ent = collections.Counter(w for row in full for w in row[4])
vocab_ent = sorted(counter_ent, key=counter_ent.get, reverse=True)

#iob_np
counter_iob_np = collections.Counter(w for row in full for w in row[5])
vocab_iob_np = sorted(counter_iob_np, key=counter_iob_np.get, reverse=True)

#iob_ner
counter_iob_ner = collections.Counter(w for row in full for w in row[6])
vocab_iob_ner = sorted(counter_iob_ner, key=counter_iob_ner.get, reverse=True)

#part_ner
counter_part_ner = collections.Counter(w for row in full for w in row[7])
vocab_part_ner = sorted(counter_part_ner, key=counter_part_ner.get, reverse=True)

#question pos
counter_q_tag = collections.Counter(w for row in full for w in row[8]) #context_tags
vocab_q_tag = sorted(counter_q_tag, key=counter_q_tag.get, reverse=True) # high rank with larger count

#question ner
counter_q_ent = collections.Counter(w for row in full for w in row[9])
vocab_q_ent = sorted(counter_q_ent, key=counter_q_ent.get, reverse=True)

#question iob_np
counter_q_iob_np = collections.Counter(w for row in full for w in row[10])
vocab_q_iob_np = sorted(counter_q_iob_np, key=counter_q_iob_np.get, reverse=True)

#question iob_ner
counter_q_iob_ner = collections.Counter(w for row in full for w in row[11])
vocab_q_iob_ner = sorted(counter_q_iob_ner, key=counter_q_iob_ner.get, reverse=True)


w2id = {w: i for i, w in enumerate(vocab)}
tag2id = {w: i for i, w in enumerate(vocab_tag)} # larger count(hight rank) with small index
ent2id = {w: i for i, w in enumerate(vocab_ent)}
iob_np2id = {w: i for i, w in enumerate(vocab_iob_np)}
iob_ner2id = {w: i for i, w in enumerate(vocab_iob_ner)}
part_ner2id = {w: i for i, w in enumerate(vocab_part_ner)}
q_tag2id = {w: i for i, w in enumerate(vocab_q_tag)}
q_ent2id = {w: i for i, w in enumerate(vocab_q_ent)}
q_iob_np2id = {w: i for i, w in enumerate(vocab_q_iob_np)}
q_iob_ner2id = {w: i for i, w in enumerate(vocab_q_iob_ner)}

log.info('Vocabulary size: {}'.format(len(vocab)))
log.info('Found {} POS tags.'.format(len(vocab_tag))) #50
log.info('Found {} entity tags: {}'.format(len(vocab_ent), vocab_ent)) #19
log.info('Found {} iob_np tags.'.format(len(vocab_iob_np))) #3
log.info('Found {} iob_ner tags.'.format(len(vocab_iob_ner))) #3
log.info('Found {} part_ner tags.'.format(len(vocab_part_ner))) #2

log.info('Found {} question POS tags.'.format(len(vocab_q_tag))) #50
log.info('Found {} question entity tags: {}'.format(len(vocab_q_tag), vocab_q_tag)) #19
log.info('Found {} question iob_np tags.'.format(len(vocab_q_iob_np))) #3
log.info('Found {} question iob_ner tags.'.format(len(vocab_q_iob_ner))) #3


def to_id(row, unk_id=1):
    context_tokens = row[1]
    context_features = row[2]
    context_tags = row[3]
    context_ents = row[4]
    context_iob_np = row[5]
    context_iob_ner = row[6]
    context_part_ner = row[7]

    q_tags = row[8]
    q_ents = row[9]
    q_iob_np = row[10]
    q_iob_ner = row[11]
    question_tokens = row[12]

    context_ids = [w2id[w] if w in w2id else unk_id for w in context_tokens]
    tag_ids = [tag2id[w] for w in context_tags]
    ent_ids = [ent2id[w] for w in context_ents]
    iob_np_ids = [iob_np2id[w] for w in context_iob_np]
    iob_ner_ids = [iob_ner2id[w] for w in context_iob_ner]
    part_ner_ids = [part_ner2id[w] for w in context_part_ner]

    question_ids = [w2id[w] if w in w2id else unk_id for w in question_tokens]
    q_tag_ids = [q_tag2id[w] for w in q_tags]
    q_ent_ids = [q_ent2id[w] for w in q_ents]
    q_iob_np_ids = [q_iob_np2id[w] for w in q_iob_np]
    q_iob_ner_ids = [q_iob_ner2id[w] for w in q_iob_ner]

    return (row[0], context_ids, context_features, tag_ids, ent_ids, iob_np_ids, iob_ner_ids, part_ner_ids,
            q_tag_ids, q_ent_ids, q_iob_np_ids, q_iob_ner_ids, question_ids) + row[13:]


train = list(map(to_id, train))
dev = list(map(to_id, dev))
log.info('converted to ids.')

# Glove emebedding
vocab_size = len(vocab)
embeddings = np.zeros((vocab_size, wv_dim))
embed_counts = np.zeros(vocab_size)
embed_counts[:2] = 1  # PADDING & UNK
with open(wv_file) as f:
    for line in f:
        elems = line.rstrip().split(' ')
        token = normalize_text(elems[0])
        if token in w2id:
            word_id = w2id[token]
            embed_counts[word_id] += 1
            embeddings[word_id] += [float(v) for v in elems[1:]] #sum glove vector for all count
embeddings /= embed_counts.reshape((-1, 1)) # take mean
log.info('got glove embedding matrix.')

# new embedding: char-level embedding
charembedding = CharEmbedding()
vocab_size = len(vocab)
char_embeddings = np.zeros((vocab_size, 100))
char_embed_counts = np.zeros(vocab_size)
char_embed_counts[:2] = 1  # PADDING & UNK
for token in w2id:
    word_id = w2id[token]
    char_embed_counts[word_id] += 1
    char_embeddings[word_id] += charembedding.emb(token)
char_embeddings /= char_embed_counts.reshape((-1, 1))
log.info('got character embedding matrix.')


# concatenate glove with charembedding to 400 dim word vector embedding layer
glove_char_embedding = np.concatenate((embeddings, char_embeddings), axis=1)
log.info('got concatenation embedding matrix.')

#------------------Save-----------------------------

meta = {
    'vocab': vocab,
    'vocab_tag': vocab_tag,
    'vocab_ent': vocab_ent,
    'embedding': embeddings.tolist(),
    'char_embedding': char_embeddings.tolist(), # in the case we need train embedding using pretrained char-level weights
    'glove_char_embedding': glove_char_embedding.tolist(),
    'vocab_iob_np': vocab_iob_np,
    'vocab_iob_ner': vocab_iob_ner,
    'vocab_part_ner': vocab_part_ner,
    'vocab_q_tag': vocab_q_tag,
    'vocab_q_ent':vocab_q_ent,
    'vocab_q_iob_np': vocab_q_iob_np,
    'vocab_q_iob_ner': vocab_q_iob_ner
}

with open('SQuAD/meta.msgpack', 'wb') as f:
    msgpack.dump(meta, f)

result = {
    'train': train,
    'dev': dev
}
# train: id, context_id, context_features, tag_id, ent_id, iob_np_ids, iob_ner_ids,part_ner_ids,
#        question_id, context, context_token_span, answer_start, answer_end
# dev:   id, context_id, context_features, tag_id, ent_id, iob_np_ids, iob_ner_ids, part_ner_ids,
#        question_id, context, context_token_span, answer
with open('SQuAD/data.msgpack', 'wb') as f:
    msgpack.dump(result, f)
if args.sample_size:
    sample = {
        'train': train[:args.sample_size],
        'dev': dev[:args.sample_size]
    }
    with open('SQuAD/sample.msgpack', 'wb') as f:
        msgpack.dump(sample, f)
log.info('saved to disk.')
