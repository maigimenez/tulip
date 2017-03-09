import numpy as np
from os.path import join, abspath, curdir, exists
from os import makedirs
from time import time
from datetime import datetime
import tensorflow as tf
from tensorflow.contrib import learn
from sklearn.model_selection import StratifiedShuffleSplit, KFold, StratifiedKFold

from corpus import Corpus
from utils import batch_iter, data_augmentation_skip
from text_cnn import TextCNN
from nin_cnn import TextNIN



def console_args():
    # Model parameters
    tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
    tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
    tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
    tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")

    # Training parameters
    tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
    tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")

    # Misc Parameters
    tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

    FLAGS = tf.flags.FLAGS
    # FLAGS.batch_size
    print("Parameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print(" - {}={}".format(attr.upper(), value))
    print("")
    return FLAGS


def default_flags():
    flags = tf.app.flags
    FLAGS = flags.FLAGS

    # Model
    # Hyperparameters
    FLAGS.embedding_dim = 300
    FLAGS.filter_sizes = '3,4,5'
    # FLAGS.filter_sizes = '3'
    FLAGS.num_filters = 100
    FLAGS.l2_reg_lambda = 3.0

    # Normalization
    FLAGS.norm = 'bn'             # {'bn', 'layer', 'none'}
    FLAGS.batch_reuse = False
    FLAGS.batch_beta = True
    FLAGS.batch_gamma = True
    FLAGS.gauss_noise = False
    FLAGS.dropout = False
    FLAGS.dropout_keep_prob = 0.5


    # Data Augmentation
    FLAGS.data_augmentation = False
    FLAGS.mult_value = 5
    FLAGS.data_augmentation_keep_prob = 0.95

    # Embeddings
    FLAGS.load_embeddings = False

    # Training parameters
    FLAGS.batch_size = 100
    FLAGS.num_epochs = 10
    FLAGS.evaluate_every = 5
    FLAGS.checkpoint_every = 5

    # If GPU is not available, it will use the CPU
    FLAGS.allow_soft_placement = True
    # Debug possible GPU related problems
    FLAGS.log_device_placement = False

    print("Default parameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print(" - {}={}".format(attr.upper(), value))
    print("")

    return FLAGS


def load_mr_data():
    MR_DATAPATH = '/home/mgimenez/Dev/corpora/MR'
    NEG_DATAPATH = join(MR_DATAPATH, 'rt-polarity.neg')
    POS_DATAPATH = join(MR_DATAPATH, 'rt-polarity.pos')
    mr_corpus = Corpus(NEG_DATAPATH, POS_DATAPATH)
    mr_corpus.load_mr()
    # print(mr_corpus.get_labels())

    return mr_corpus


def build_vocabulary(texts):
    """" Build vocabulary, the lookup table and transform the text """
    max_document_length = max([len(x.split(" ")) for x in texts])
    # Creates the lookup table: Maps documents to sequences of word ids.
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    text_lookup = np.array(list(vocab_processor.fit_transform(texts)))
    # print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))

    return text_lookup, vocab_processor


def split_train_test(text_lookup, labels, nfolds=None):
    # unique_labels = [label[0] for label in labels]
    if nfolds:
        k_folds = KFold(n_splits=nfolds, shuffle=True, random_state=0)
        # k_folds = StratifiedShuffleSplit(n_splits=nfolds, test_size=0.2, random_state=0)
        # k_folds = StratifiedKFold(n_splits=nfolds, random_state=0)
        for train_idx, test_idx in k_folds.split(text_lookup, labels):
            yield text_lookup[train_idx], text_lookup[test_idx], labels[train_idx], labels[test_idx]
    else:
        # Randomly shuffle data
        np.random.seed(10)
        shuffle_indices = np.random.permutation(np.arange(len(labels)))
        x_shuffled = text_lookup[shuffle_indices]
        y_shuffled = labels[shuffle_indices]

        # Split train/test set
        x_train, x_dev = x_shuffled[:-1000], x_shuffled[-1000:]
        y_train, y_dev = y_shuffled[:-1000], y_shuffled[-1000:]
        print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

        yield x_train, x_dev, y_train, y_dev


def load_word_embedding(vocabulary, embedding_size):
    init_embedding = np.random.uniform(-1.0, 1.0, (len(vocabulary), embedding_size))
    # Read W2V
    w2v_path = "/home/mgimenez/Dev/corpora/w2v/GoogleNews-vectors-negative300.bin"
    with open(w2v_path, "rb") as f:
        header = f.readline()
        vocab_size, embedding_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * embedding_size
        for num_word in range(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == b' ':
                    break
                elif ch != b'\n':  # ignore newlines in front of words (some binary files have)
                    word.append(ch)
            word = str(b''.join(word), encoding="latin-1", errors="strict")
            vector_embedding = np.fromstring(f.read(binary_len), dtype='float32')
            idx_word = vocabulary.get(word)
            # If the word from w2v exists in the current dataset dataset add it.
            if idx_word:
                init_embedding[idx_word] = vector_embedding
    return init_embedding


def train_step(FLAGS, sess, cnn, x_batch, y_batch, train_op, global_step,
               train_summary_op, batch_step, train_summary_writer):
    """
    A single training step
    """
    feed_dict = {
        cnn.input_x: x_batch,
        cnn.input_y: y_batch,
        cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
        # cnn.dropout_keep_prob: 1.0,
        cnn.is_training: True

    }
    _, step, summaries, loss, accuracy = sess.run(
        [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
        feed_dict)
    time_str = datetime.now().isoformat()
    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
    train_summary_writer.add_summary(summaries, step)


def dev_step(sess, cnn, x_batch, y_batch, global_step, dev_summary_op, batch_step, writer=None):
    """
    Evaluates model on a dev set
    """
    feed_dict = {
        cnn.input_x: x_batch,
        cnn.input_y: y_batch,
        cnn.dropout_keep_prob: 1.0,
        cnn.is_training: False
    }
    step, summaries, loss, accuracy = sess.run(
        [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
        feed_dict)
    time_str = datetime.now().isoformat()
    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, batch_step, loss, accuracy))
    if writer:
        writer.add_summary(summaries, step)
    return accuracy


def train_fold(config_flags, vocab_processor, x_train, x_dev, y_train, y_dev, init_embeddings=None):
    # TODO allow console based parameters
    # console_FLAGS = console_args() otherwise default_flags()
    FLAGS = config_flags

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextNIN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                kernel_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_kernels=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                norm=FLAGS.norm,
                batch_reuse=FLAGS.batch_reuse,
                batch_beta=FLAGS.batch_beta,
                batch_gamma=FLAGS.batch_gamma,
                gauss_noise=FLAGS.gauss_noise,
                dropout=FLAGS.dropout)

            print(" - VOCABULARY SIZE ={}".format(len(vocab_processor.vocabulary_)))
            print(cnn.input_x.get_shape())

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            #optimizer = tf.train.AdamOptimizer(1e-3)
            optimizer = tf.train.AdamOptimizer(0.2)
            # optimizer = tf.train.AdadeltaOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time()))
            out_dir = abspath(join(curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)
            # embeddings_summary = tf.scalar_summary("embeddings", cnn.embedded_chars)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = abspath(join(out_dir, "checkpoints"))
            checkpoint_prefix = join(checkpoint_dir, "model")
            if not exists(checkpoint_dir):
                makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables())

            # Write vocabulary
            vocab_processor.save(join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # Load embeddings
            if init_embeddings is not None:
                sess.run(cnn.W_embedding.assign(init_embeddings))

            # Generate batches
            data = list(zip(x_train, y_train))
            batch_size = FLAGS.batch_size
            num_batches_per_epoch = int(len(data) / batch_size) + 1
            num_train_batches = int(np.round(num_batches_per_epoch * 0.9))
            num_val_batches = num_batches_per_epoch - num_train_batches
            batches = batch_iter(data, batch_size, FLAGS.num_epochs, num_batches_per_epoch)
            best_val_accuracy = 0
            val_accuracies = []
            cur_epoch, test_epoch = 1, None
            print(num_batches_per_epoch, num_train_batches, num_val_batches)
            for cur_batch, batch in enumerate(batches):
                x_batch, y_batch = zip(*batch)

                if cur_batch < ((cur_epoch - 1) * num_batches_per_epoch) + num_train_batches:
                    # print('train step ', cur_batch)
                    current_step = tf.train.global_step(sess, global_step)
                    train_step(FLAGS, sess, cnn, x_batch, y_batch, train_op, global_step, train_summary_op, current_step, train_summary_writer)

                elif cur_batch < (cur_epoch * num_batches_per_epoch):
                    print("Validation: ")
                    val_accuracy = dev_step(sess, cnn, x_batch, y_batch, global_step, dev_summary_op, cur_batch, writer=dev_summary_writer)
                    val_accuracies.append(val_accuracy)
                else:
                    print('\nEarly stop:', cur_batch)
                    cur_epoch += 1
                    mean_val_accuracies = np.mean(val_accuracies)
                    print('\n({}) Early stop: current mean:{} -  best: {}'.format(cur_batch, mean_val_accuracies, best_val_accuracy))
                    if mean_val_accuracies >= best_val_accuracy:
                        best_val_accuracy = mean_val_accuracies
                        print('Validate test in epoch {}'.format(cur_epoch))
                        test_accuracy = dev_step(sess, cnn, x_dev, y_dev, global_step, dev_summary_op, cur_batch, writer=dev_summary_writer)
                        test_epoch = cur_batch
                        print()

                    val_accuracies = []

            last_accuracy = dev_step(sess, cnn, x_dev, y_dev, global_step, dev_summary_op, cur_batch, writer=dev_summary_writer)
            if last_accuracy >= test_accuracy:
                print('------------------------------------> last')
                return last_accuracy
            return test_accuracy


def train(config_flags, vocab_processor, x_train, y_train, x_dev, y_dev, x_test, y_test, init_embeddings=None):
    # TODO allow console based parameters
    # console_FLAGS = console_args() otherwise default_flags()
    FLAGS = config_flags

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextNIN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                kernel_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_kernels=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                norm=FLAGS.norm,
                batch_reuse=FLAGS.batch_reuse,
                batch_beta=FLAGS.batch_beta,
                batch_gamma=FLAGS.batch_gamma,
                gauss_noise=FLAGS.gauss_noise,
                dropout=FLAGS.dropout)

            print(" - VOCABULARY SIZE ={}".format(len(vocab_processor.vocabulary_)))
            print(cnn.input_x.get_shape())

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            # optimizer = tf.train.AdadeltaOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time()))
            out_dir = abspath(join(curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)
            # embeddings_summary = tf.scalar_summary("embeddings", cnn.embedded_chars)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = join(out_dir, "summaries", "train")
            #train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = abspath(join(out_dir, "checkpoints"))
            checkpoint_prefix = join(checkpoint_dir, "model")
            if not exists(checkpoint_dir):
                makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables())

            # Write vocabulary
            vocab_processor.save(join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # Load embeddings
            if init_embeddings is not None:
                sess.run(cnn.W_embedding.assign(init_embeddings))

            # Generate batches
            data = list(zip(x_train, y_train))
            batch_size = FLAGS.batch_size
            print(len(data), batch_size)
            num_batches_per_epoch = int(len(data) / batch_size) + 1

            batches = batch_iter(data, batch_size, FLAGS.num_epochs, num_batches_per_epoch)
            best_val_accuracy = 0
            val_accuracies = []
            cur_epoch, test_epoch = 1, None
            print(num_batches_per_epoch, len(x_dev), len(x_test))
            for cur_batch, batch in enumerate(batches):
                x_batch, y_batch = zip(*batch)
                if cur_batch < (cur_epoch * num_batches_per_epoch):
                    current_step = tf.train.global_step(sess, global_step)
                    train_step(FLAGS, sess, cnn, x_batch, y_batch, train_op, global_step, train_summary_op,
                               current_step, train_summary_writer)
                else:
                    cur_epoch += 1
                    val_accuracy = dev_step(sess, cnn, x_dev, y_dev, global_step, dev_summary_op, cur_batch,
                                            writer=dev_summary_writer)
                    print("TEST")
                    if val_accuracy >= best_val_accuracy:
                        best_val_accuracy = val_accuracy
                        print('Validate test in epoch {}'.format(cur_epoch))
                        test_accuracy = dev_step(sess, cnn, x_test, y_test, global_step, dev_summary_op, cur_batch, writer=dev_summary_writer)
                        test_epoch = cur_batch
                        print()
            last_accuracy = dev_step(sess, cnn, x_test, y_test, global_step, dev_summary_op, cur_batch, writer=dev_summary_writer)
            if last_accuracy >= test_accuracy:
                print('------------------------------------> last')
                return last_accuracy
            return test_accuracy


def load_mr_data():
    MR_DATAPATH = '/home/mgimenez/Dev/corpora/MR'
    NEG_DATAPATH = join(MR_DATAPATH, 'rt-polarity.neg')
    POS_DATAPATH = join(MR_DATAPATH, 'rt-polarity.pos')
    mr_corpus = Corpus(NEG_DATAPATH, POS_DATAPATH)
    mr_corpus.load_mr()
    # print(mr_corpus.get_labels())

    return mr_corpus


def sst_experiment():
    # Load the data sets
    SST_PATH = "/home/mgimenez/Dev/corpora/SST/csvs"
    TRAIN_PATH = join(SST_PATH, "train_dataset.csv")
    TEST_PATH = join(SST_PATH, "test_dataset.csv")
    DEV_PATH = join(SST_PATH, "dev_dataset.csv")

    train_corpus = Corpus(TRAIN_PATH)
    train_corpus.load_sst()
    test_corpus = Corpus(TEST_PATH)
    test_corpus.load_sst()
    dev_corpus = Corpus(DEV_PATH)
    dev_corpus.load_sst()

    # Validate that you have load the data properly
    assert len(train_corpus.data) == 8544
    assert len(test_corpus.data) == 2210
    assert len(dev_corpus.data) == 1101
    print(len(train_corpus.data), len(test_corpus.data), len(dev_corpus.data))

    train_sentences = train_corpus.get_texts()
    train_labels = train_corpus.get_labels()
    test_sentences = test_corpus.get_texts()
    test_labels = test_corpus.get_labels()
    dev_sentences = dev_corpus.get_texts()
    dev_labels = dev_corpus.get_labels()

    # Create the vocabulary
    text_lookup_train, vocab_processor = build_vocabulary(train_sentences)
    text_lookup_test = np.array(list(vocab_processor.fit_transform(test_sentences)))
    dev_lookup_test = np.array(list(vocab_processor.fit_transform(dev_sentences)))
    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))

    # Load the flags
    FLAGS = default_flags()

    # Load the word embeddings if required
    if FLAGS.load_embeddings:
        print("* With embeddings")
        init_embedding = load_word_embedding(vocab_processor.vocabulary_, FLAGS.embedding_dim)
    else:
        print("* WithOUT embeddings")
        init_embedding = None


    accuracy = train(FLAGS, vocab_processor, text_lookup_train, train_labels,
                     dev_lookup_test, dev_labels,
                     text_lookup_test, test_labels, init_embeddings=None)
    print(accuracy)


def mr_experiment():
    mr_corpus = load_mr_data()
    texts = mr_corpus.get_texts()
    labels = mr_corpus.get_labels()
    text_lookup, vocab_processor = build_vocabulary(texts)
    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    # # x_train, x_dev, y_train, y_dev = next(split_train_test(text_lookup, labels))
    FLAGS = default_flags()

    if FLAGS.load_embeddings:
        print("* With embeddings")
        init_embedding = load_word_embedding(vocab_processor.vocabulary_, FLAGS.embedding_dim)
    else:
        print("* WithOUT embeddings")
        init_embedding = None

    NUM_FOLDS = 10
    accuracies = np.zeros(NUM_FOLDS)
    current_fold = 0
    for x_train, x_dev, y_train, y_dev in split_train_test(text_lookup, labels, NUM_FOLDS):
        print("Train/Dev split: {:d}/{:d}".format(len(x_train), len(x_dev)))
        print(x_train.shape)
        if FLAGS.data_augmentation:
            x_train = data_augmentation_skip(x_train, keep_prob=FLAGS.data_augmentation_keep_prob,
                                                mul_value=FLAGS.mult_value)
            print(x_train.shape)

        accuracies[current_fold] = train_fold(FLAGS, vocab_processor,
                                              x_train, x_dev, y_train, y_dev, init_embedding)
        current_fold += 1
        print("")

    print(accuracies)
    print(accuracies.mean())
    print(accuracies.std())

if __name__ == "__main__":
    sst_experiment()
    # mr_experiment

