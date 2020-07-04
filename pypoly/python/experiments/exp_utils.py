import pdb

import random
random.seed(12345)

__all__ = [
    "as_tensor",
    "gen_random_dataset",
]

from xtype import MutableTensor
from xtype import ImmutableArray


def as_tensor(x, mutable=True):
    """wrap numpy's ndarray into a Tensor."""
    if mutable:
        return MutableTensor(x.shape, x.dtype, value=x)
    else:
        return ImmuatableTensor(x.shape, x.dtype, value=x)


def random_int(min_v, max_v, n):
    random_list = []
    for i in range(n):
        random_list.append(random.randint(1, 30))
    return random_list


def gen_random_dataset():
    max_word_num = 20

    min_sentence_num = 2
    max_sentence_num = 6

    min_seq_len = 2
    max_seq_len = 10

    min_batch_size = 1
    max_batch_size = 5

    # Level 0: batch of sentences in a passage
    sentence_num = 5

    max_batch_size = 5
    max_batch_num = 3

    batches = []
    batch_num = random.randint(1, max_batch_num)
    print(f'{batch_num} batches is generated.')
    for i in range(batch_num):

        passages = []
        batch_size = random.randint(2, max_batch_size)

        print(f'batch {i} has {batch_size} passages.')
        for j in range(batch_size):

            sentence_num = random.randint(min_sentence_num, max_sentence_num)
            print(f'  |__ passage {j} has {sentence_num} sentences.')

            sentences = []
            for k in range(sentence_num):
                word_num = random.randint(min_seq_len, max_seq_len)

                print(f'    |__ sentence {k} has {word_num} words.')
                words = ImmutableArray(
                    [random.randint(0, max_word_num) for _ in range(word_num)])
                sentences.append(words)
            passages.append(ImmutableArray(sentences))
        batches.append(ImmutableArray(passages))
    return ImmutableArray(batches)


def variable_len_seq_batch(batch_size=16, max_seq_len=20, max_words=50):
    batch = []
    for i in range(batch_size):
        seq_len = random.randint(1, max_seq_len)
        batch.append([random.randint(1, max_words) for _ in range(seq_len)])
    return batch
