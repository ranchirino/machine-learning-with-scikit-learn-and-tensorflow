import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


#%% Fetch the data
from six.moves import urllib

import errno
import os
import zipfile

WORDS_PATH = "datasets/words"
WORDS_URL = 'http://mattmahoney.net/dc/text8.zip'

def mkdir_p(path):
    """Create directories, ok if they already exist.

    This is for python 2 support. In python >=3.2, simply use:
    >>> os.makedirs(path, exist_ok=True)
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def fetch_words_data(words_url=WORDS_URL, words_path=WORDS_PATH):
    os.makedirs(words_path, exist_ok=True)
    zip_path = os.path.join(words_path, "words.zip")
    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(words_url, zip_path)
    with zipfile.ZipFile(zip_path) as f:
        data = f.read(f.namelist()[0])
    return data.decode("ascii").split()

words = fetch_words_data()

words[:5]
# Out[4]: ['anarchism', 'originated', 'as', 'a', 'term']

#%% Build the dictionary
from collections import Counter

vocabulary_size = 50000

vocabulary = [("UNK", None)] + Counter(words).most_common(vocabulary_size - 1)
vocabulary = np.array([word for word, _ in vocabulary])
dictionary = {word: code for code, word in enumerate(vocabulary)}
data = np.array([dictionary.get(word, 0) for word in words])

" ".join(words[:9]), data[:9]
# Out[10]:
# ('anarchism originated as a term of abuse first used',
#  array([5234, 3081,   12,    6,  195,    2, 3134,   46,   59]))

" ".join([vocabulary[word_index] for word_index in [5241, 3081, 12, 6, 195, 2, 3134, 46, 59]])
# Out[11]: 'cycles originated as a term of abuse first used'

words[24], data[24]
# Out[12]: ('culottes', 0)


#%% Generate batches
import random
from collections import deque

def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1 # [ skip_window target skip_window ]
    buffer = deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [ skip_window ]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels

data_index=0
batch, labels = generate_batch(8, 2, 1)

batch, [vocabulary[word] for word in batch]
# Out[15]:
# (array([3081, 3081,   12,   12,    6,    6,  195,  195]),
#  ['originated', 'originated', 'as', 'as', 'a', 'a', 'term', 'term'])

labels, [vocabulary[word] for word in labels[:, 0]]
# Out[16]:
# (array([[5234],
#         [  12],
#         [   6],
#         [3081],
#         [ 195],
#         [  12],
#         [   2],
#         [   6]]),
#  ['anarchism', 'as', 'a', 'originated', 'term', 'as', 'of', 'a'])


#%% Build the model
batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # Number of negative examples to sample.

learning_rate = 0.01

reset_graph()

# Input data.
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

vocabulary_size = 50000
embedding_size = 150

# Look up embeddings for inputs.
init_embeds = tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)
embeddings = tf.Variable(init_embeds)

train_inputs = tf.placeholder(tf.int32, shape=[None])
embed = tf.nn.embedding_lookup(embeddings, train_inputs)

# Construct the variables for the NCE loss
nce_weights = tf.Variable(
    tf.truncated_normal([vocabulary_size, embedding_size],
                        stddev=1.0 / np.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

# Compute the average NCE loss for the batch.
# tf.nce_loss automatically draws a new sample of the negative labels each
# time we evaluate the loss.
loss = tf.reduce_mean(
    tf.nn.nce_loss(nce_weights, nce_biases, train_labels, embed,
                   num_sampled, vocabulary_size))

# Construct the Adam optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

# Compute the cosine similarity between minibatch examples and all embeddings.
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), axis=1, keep_dims=True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

# Add variable initializer.
init = tf.global_variables_initializer()


#%% Train the model
num_steps = 10001

with tf.Session() as session:
    init.run()

    average_loss = 0
    for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}

        # We perform one update step by evaluating the training op (including it
        # in the list of returned values for session.run()
        _, loss_val = session.run([training_op, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print("\rIteration: {}".format(step), end="\t")
            print("Average loss at step ", step, ": ", average_loss)
            average_loss = 0

        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = vocabulary[valid_examples[i]]
                top_k = 8 # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k+1]
                log_str = "Nearest to %s:" % valid_word
                for k in range(top_k):
                    close_word = vocabulary[nearest[k]]
                    log_str = "%s %s," % (log_str, close_word)
                print(log_str)

    final_embeddings = normalized_embeddings.eval()

# Iteration: 0	Average loss at step  0 :  294.8122863769531
# Nearest to may: failure, rna, efficacious, aspirin, lecompton, definitive, geese, amphibious,
# Nearest to were: bezier, antibodies, nicknamed, panthers, compiler, tao, smarter, busy,
# Nearest to nine: cocker, struck, dame, tensile, unifies, operetta, develop, opinions,
# Nearest to over: tt, tuned, manichaeans, fractional, cambridge, balaguer, fluoride, strenuously,
# Nearest to a: unwavering, delson, luisa, uan, rotterdam, apologised, enciclopedia, openly,
# Nearest to all: redevelopment, mailbox, proposing, laika, predicting, raceway, knoxville, reuters,
# Nearest to history: eliot, dbu, grimes, andrey, minicomputer, fiume, greenfacts, conclusions,
# Nearest to during: reactor, vultures, nuts, intuitions, trypanosomiasis, welcomed, clustered, rial,
# Nearest to UNK: reykjav, fi, rosalyn, mainline, archaeologist, ean, armstrong, stevenage,
# Nearest to with: devlin, main, coexist, hernando, backronym, bhangra, hopper, hamming,
# Nearest to will: commerce, slow, attested, obsession, christy, coursework, secretive, ballroom,
# Nearest to only: desperation, harmed, tanaka, drown, alkenes, candu, laughter, illustrating,
# Nearest to while: ejected, unsound, cheney, zimbabwean, falsely, irr, worrying, forget,
# Nearest to these: pedro, condoned, neck, ssn, supervising, doug, thereto, melton,
# Nearest to such: accessing, roz, futurists, already, dmt, mda, pauly, rask,
# Nearest to often: presses, convenience, arabidopsis, jewry, mapuche, guts, macrinus, implemented,

# Iteration: 2000	Average loss at step  2000 :  131.21949031543733
# Iteration: 4000	Average loss at step  4000 :  62.389544053316115
# Iteration: 6000	Average loss at step  6000 :  42.05166925835609
# Iteration: 8000	Average loss at step  8000 :  30.956905294179915

# Iteration: 10000	Average loss at step  10000 :  26.223095872044563
# Nearest to may: are, must, is, nitroglycerin, atomism, archaelogical, that, arif,
# Nearest to were: was, asks, ions, durrani, murchison, recovered, sugarcane, alexandre,
# Nearest to nine: one, eight, seven, three, zero, six, four, five,
# Nearest to over: reversal, subproblems, spectral, diverged, daimler, stadium, calais, lacan,
# Nearest to a: the, and, morpork, this, aquarius, is, canola, horch,
# Nearest to all: actinium, uterus, diminished, ufos, afc, hermetical, altaic, the,
# Nearest to history: nsu, editions, clerical, conformations, charity, respondent, dialect, exporter,
# Nearest to during: purple, subproblems, illyrians, specified, altaic, guard, vineyards, tops,
# Nearest to UNK: one, the, five, two, seven, and, of, six,
# Nearest to with: of, and, altaic, in, for, seven, ataxia, as,
# Nearest to will: explain, levitt, could, underpinnings, baskets, dependencies, cooperstown, owe,
# Nearest to only: hybrids, companionship, nf, mogadishu, barbuda, isthmus, it, durrani,
# Nearest to while: in, tarleton, altaic, adept, actinium, antisymmetric, of, peanut,
# Nearest to these: genes, appeasement, mpla, it, provisions, everywhere, actinium, informed,
# Nearest to such: terrace, of, cambrian, alp, excess, emmer, mileva, mislead,
# Nearest to often: not, a, nonaggression, retroviral, was, schopenhauer, altaic, zf,

# Let's save the final embeddings (of course you can use a TensorFlow Saver if you prefer):
np.save("./my_final_embeddings.npy", final_embeddings)


#%% Plot the embeddings
def plot_with_labels(low_dim_embs, labels):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  #in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i,:]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

from sklearn.manifold import TSNE

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
plot_only = 500
low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only,:])
labels = [vocabulary[i] for i in range(plot_only)]
plot_with_labels(low_dim_embs, labels)


#%% Machine Translation
# The basic_rnn_seq2seq() function creates a simple Encoder/Decoder model:
# it first runs an RNN to encode encoder_inputs into a state vector,
# then runs a decoder initialized with the last encoder state on decoder_inputs.
# Encoder and decoder use the same RNN cell type but they don't share parameters.

reset_graph()

n_steps = 50
n_neurons = 200
n_layers = 3
num_encoder_symbols = 20000
num_decoder_symbols = 20000
embedding_size = 150
learning_rate = 0.01

X = tf.placeholder(tf.int32, [None, n_steps]) # English sentences
Y = tf.placeholder(tf.int32, [None, n_steps]) # French translations
W = tf.placeholder(tf.float32, [None, n_steps - 1, 1])
Y_input = Y[:, :-1]
Y_target = Y[:, 1:]

encoder_inputs = tf.unstack(tf.transpose(X)) # list of 1D tensors
decoder_inputs = tf.unstack(tf.transpose(Y_input)) # list of 1D tensors

lstm_cells = [tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons)
              for layer in range(n_layers)]
cell = tf.contrib.rnn.MultiRNNCell(lstm_cells)

output_seqs, states = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
    encoder_inputs,
    decoder_inputs,
    cell,
    num_encoder_symbols,
    num_decoder_symbols,
    embedding_size)

logits = tf.transpose(tf.unstack(output_seqs), perm=[1, 0, 2])

logits_flat = tf.reshape(logits, [-1, num_decoder_symbols])
Y_target_flat = tf.reshape(Y_target, [-1])
W_flat = tf.reshape(W, [-1])
xentropy = W_flat * tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y_target_flat, logits=logits_flat)
loss = tf.reduce_mean(xentropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

