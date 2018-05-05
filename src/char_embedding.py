# pip install embeddings
from embeddings.embedding import Embedding
import numpy as np

def ngrams(sentence, n):
    """
    Returns:
        list: a list of lists of words corresponding to the ngrams in the sentence.
    """
    return [sentence[i:i+n] for i in range(len(sentence)-n+1)]

class CharEmbedding(Embedding):

    size = 874474
    d_emb = 100

    def __init__(self):

        self.db = self.initialize_db(self.path('char/kazuma.db'))

        if len(self) < self.size:
            self.clear()
            self.load_word2emb()

    def emb(self, w, default='zero'):
        assert default == 'zero', 'only zero default is supported for character embeddings'
        chars = ['#BEGIN#'] + list(w) + ['#END#']
        embs = np.zeros(self.d_emb, dtype=np.float32)
        match = {}
        for i in [2, 3, 4]:
            grams = ngrams(chars, i)
            for g in grams:
                g = '{}gram-{}'.format(i, ''.join(g))
                e = self.lookup(g)
                if e is not None:
                    match[g] = np.array(e, np.float32)
        if match:
            embs = sum(match.values()) / len(match)
        return embs.tolist()

    def load_word2emb(self, batch_size=1000):
        seen = set()
        fin_name = 'char/charNgram.txt'
        with open(fin_name, 'r') as ftxt:
            content = ftxt.read()
            lines = content.splitlines()
            batch = []
            for line in lines:
                elems = line.rstrip().split()
                vec = [float(n) for n in elems[-d_emb:]]
                word = ' '.join(elems[:-d_emb])
                if word in seen:
                    continue
                seen.add(word)
                batch.append((word, vec))
                if len(batch) == batch_size:
                    self.insert_batch(batch)
                    batch.clear()
            if batch:
                self.insert_batch(batch)
