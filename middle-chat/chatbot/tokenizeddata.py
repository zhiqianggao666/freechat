# Copyright 2017 Bo Shao. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import codecs
import os
import tensorflow as tf

from collections import namedtuple
from tensorflow.python.ops import lookup_ops
from tensorflow.python.platform import gfile
from chatbot.hparams import HParams
import unicodedata
import re
import jieba
import numpy as np

AUG0_FOLDER = "Augment0"

VOCAB_FILE = "vocab.txt"


def split_chinese(sentence):
    line = re.split(u'【（。，！？、“：；）】', sentence.strip())
    ws = []
    for words in line:
        for w in words:
            ws.append(w)
    return ws

def create_vocab(file_dir, vocab_file,hparams):
    file_list = []
    for data_file in sorted(os.listdir(file_dir)):
        full_path_name = os.path.join(file_dir, data_file)
        if os.path.isfile(full_path_name) and data_file.lower().endswith('.conv'):
            file_list.append(full_path_name)
    
    assert len(file_list) > 0
    if not gfile.Exists(vocab_file):
        print("Creating vocabulary-------:%s", vocab_file)
        vocab = {}
        for fp in file_list:
            print('fp-------%s', fp)
            with gfile.GFile(fp, mode="rb") as f:
                counter = 0
                for line in f:
                    if counter > hparams.example_num:
                        break
                    counter += 1
                    line = tf.compat.as_bytes(line)
                    decoded_str = line.decode('utf-8')
                    if counter % 100000 == 0:
                        print("  processing line %d" % counter)
                    if decoded_str.startswith('M') and len(decoded_str) > 3:
                        tokens = split_chinese(decoded_str[2:-1])
                        for w in tokens:
                            word = w
                            if word in vocab:
                                vocab[word] += 1
                            else:
                                vocab[word] = 1
        vocab_list = [hparams.pad_token,hparams.bos_token,hparams.eos_token,hparams.unk_token] + sorted(vocab, key=vocab.get, reverse=True)
        
        print('>> Full Vocabulary Size :', len(vocab_list))
        if len(vocab_list) > hparams.vocab_size:
            vocab_list = vocab_list[:hparams.vocab_size]
        with gfile.GFile(vocab_file, mode="wb") as vf:
            for w in vocab_list:
                if type(w) is str:
                    words = bytes(w.encode('utf-8')) + b"\n"
                    vf.write(words)
                else:
                    words = w + b"\n"
                    vf.write(words)

    
class TokenizedData:
    def __init__(self, corpus_dir, hparams=None, training=True, buffer_size=8192):
        """
        Args:
            corpus_dir: Name of the folder storing corpus files for training.
            hparams: The object containing the loaded hyper parameters. If None, it will be 
                    initialized here.
            training: Whether to use this object for training.
            buffer_size: The buffer size used for mapping process during data processing.
        """
        if hparams is None:
            self.hparams = HParams(corpus_dir).hparams
        else:
            self.hparams = hparams

        self.src_max_len = self.hparams.src_max_len
        self.tgt_max_len = self.hparams.tgt_max_len

        self.training = training
        self.text_set = None
        self.id_set = None
        vocab_file = os.path.join(corpus_dir, VOCAB_FILE)
        file_dir = os.path.join(corpus_dir, AUG0_FOLDER)
        create_vocab(file_dir, vocab_file, self.hparams)
        self.vocab_size, _ = check_vocab(vocab_file)
        self.vocab_table = lookup_ops.index_table_from_file(vocab_file,
                                                            default_value=self.hparams.unk_id)

        if training:
            self.reverse_vocab_table = None
            self._load_corpus(corpus_dir)
            self._convert_to_tokens(buffer_size)
        else:
            self.reverse_vocab_table = \
                lookup_ops.index_to_string_table_from_file(vocab_file,
                                                           default_value=self.hparams.unk_token)
        

        

    def get_training_batch(self, num_threads=4):
        assert self.training

        buffer_size = self.hparams.batch_size * 400

        # Comment this line for debugging.
        #train_set = self.id_set.shuffle(buffer_size=buffer_size)

        # Create a target input prefixed with BOS and a target output suffixed with EOS.
        # After this mapping, each element in the train_set contains 3 columns/items.
        train_set = self.id_set.map(lambda src, tgt:
                                  (src, tf.concat(([self.hparams.bos_id], tgt), 0),
                                   tf.concat((tgt, [self.hparams.eos_id]), 0)),
                                  num_parallel_calls=num_threads).prefetch(buffer_size)


        # Add in sequence lengths.
        train_set = train_set.map(lambda src, tgt_in, tgt_out:
                                  (src, tgt_in, tgt_out, tf.size(src), tf.size(tgt_in)),
                                  num_parallel_calls=num_threads).prefetch(buffer_size)
        
        '''
        dataset = train_set
        dataset = dataset.batch(1)
        iter = dataset.make_initializable_iterator()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            sess.run(iter.initializer)
            a = sess.run(iter.get_next())
            print(a[0])
            print(a[1])
            a = 1
        '''
        
            
        def batching_func(x):
            return x.padded_batch(
                self.hparams.batch_size,
                # The first three entries are the source and target line rows, these have unknown-length
                # vectors. The last two entries are the source and target row sizes, which are scalars.
                padded_shapes=(tf.TensorShape([None]),  # src
                               tf.TensorShape([None]),  # tgt_input
                               tf.TensorShape([None]),  # tgt_output
                               tf.TensorShape([]),      # src_len
                               tf.TensorShape([])),     # tgt_len
                # Pad the source and target sequences with eos tokens. Though we don't generally need to
                # do this since later on we will be masking out calculations past the true sequence.
                padding_values=(self.hparams.eos_id,  # src
                                self.hparams.eos_id,  # tgt_input
                                self.hparams.eos_id,  # tgt_output
                                0,       # src_len -- unused
                                0))      # tgt_len -- unused

        if self.hparams.num_buckets > 1:
            bucket_width = (self.src_max_len + self.hparams.num_buckets - 1) // self.hparams.num_buckets

            # Parameters match the columns in each element of the dataset.
            def key_func(unused_1, unused_2, unused_3, src_len, tgt_len):
                # Calculate bucket_width by maximum source sequence length. Pairs with length [0, bucket_width)
                # go to bucket 0, length [bucket_width, 2 * bucket_width) go to bucket 1, etc. Pairs with
                # length over ((num_bucket-1) * bucket_width) words all go into the last bucket.
                # Bucket sentence pairs by the length of their source sentence and target sentence.
                bucket_id = tf.maximum(src_len // bucket_width, tgt_len // bucket_width)
                return tf.to_int64(tf.minimum(self.hparams.num_buckets, bucket_id))

            # No key to filter the dataset. Therefore the key is unused.
            def reduce_func(unused_key, windowed_data):
                return batching_func(windowed_data)

            batched_dataset = train_set.apply(
                tf.contrib.data.group_by_window(key_func=key_func,
                                                reduce_func=reduce_func,
                                                window_size=self.hparams.batch_size))
        else:
            batched_dataset = batching_func(train_set)

        batched_iter = batched_dataset.make_initializable_iterator()
        
        '''
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            sess.run(batched_iter.initializer)
            a = sess.run(batched_iter.get_next())
            print(a[0])
            print(a[1])
            print(a[2])
            print(a[3])
            print(a[4])
            a = 1
        '''
            
        (src_ids, tgt_input_ids, tgt_output_ids, src_seq_len, tgt_seq_len) = (batched_iter.get_next())

        return BatchedInput(initializer=batched_iter.initializer,
                            source=src_ids,
                            target_input=tgt_input_ids,
                            target_output=tgt_output_ids,
                            source_sequence_length=src_seq_len,
                            target_sequence_length=tgt_seq_len)

    def map_pyfunct_infer(self, src):
        return [word for word in src.decode()], 1

    def map_split_func_infer(self, src):
        a = tf.py_func(self.map_pyfunct_infer, [src], [tf.string, tf.int64])
        return a
    
    def print_data(self):
        id_dataset = self.src_dataset.map(lambda src: tf.cast(self.vocab_table.lookup(src), tf.int32))
        return id_dataset
    
    def get_inference_batch(self, src_dataset):
        self.src_dataset = src_dataset
        #text_dataset = src_dataset.map(self.map_split_func_infer)
        #text_dataset = src_dataset.map(lambda src:tf.string_split(src))

        id_dataset = src_dataset.map(lambda src:  tf.cast(self.vocab_table.lookup(src), tf.int32))
        
        if self.hparams.src_max_len_infer:
            id_dataset = id_dataset.map(lambda src: src[:self.hparams.src_max_len_infer])

        if self.hparams.source_reverse:
            id_dataset = id_dataset.map(lambda src: tf.reverse(src, axis=[0]))
            
        # Add in the word counts.
        id_dataset = id_dataset.map(lambda src: (src, tf.size(src)))

        def batching_func(x):
            return x.padded_batch(
                self.hparams.batch_size_infer,
                # The entry is the source line rows; this has unknown-length vectors.
                # The last entry is the source row size; this is a scalar.
                padded_shapes=(tf.TensorShape([None]),  # src
                               tf.TensorShape([])),     # src_len
                # Pad the source sequences with eos tokens. Though notice we don't generally need to
                # do this since later on we will be masking out calculations past the true sequence.
                padding_values=(self.hparams.eos_id,  # src
                                0))                   # src_len -- unused

        id_dataset = batching_func(id_dataset)

        infer_iter = id_dataset.make_initializable_iterator()
        (src_ids, src_seq_len) = infer_iter.get_next()

        return BatchedInput(initializer=infer_iter.initializer,
                            source=src_ids,
                            target_input=None,
                            target_output=None,
                            source_sequence_length=src_seq_len,
                            target_sequence_length=None)

    def _load_corpus(self, corpus_dir):
        file_list = []
        file_dir = os.path.join(corpus_dir, AUG0_FOLDER)

        for data_file in sorted(os.listdir(file_dir)):
            full_path_name = os.path.join(file_dir, data_file)
            if os.path.isfile(full_path_name) and data_file.lower().endswith('.conv'):
                file_list.append(full_path_name)

        assert len(file_list) > 0
        src_data=[]
        tgt_data=[]
        for fp in file_list:
            with gfile.GFile(fp, mode="rb") as f:
                counter = 0
                conversation = []
                for line in f:
                    if counter > self.hparams.example_num:
                        break
                    line = tf.compat.as_bytes(line)
                    counter += 1
                    if counter % 100000 == 0:
                        print("  processing line %d" % counter)
                    decoded_str=line.decode('utf-8')
                    #if counter  == 8291:
                    #    print(" decoded_str : %s" , decoded_str)
                    if decoded_str.startswith('M')  and len(decoded_str) > 3:
                        conversation.append(decoded_str)
                    else:
                        if decoded_str.startswith('E')  and len(decoded_str) <= 2:
                            if len(conversation) % 2 != 0:
                                conversation=conversation[:-1]
                            for i,each_chat in enumerate(conversation):
                                if i % 2 == 0:
                                    src_data.append(list(each_chat[2:-1]))
                                else:
                                    tgt_data.append(list(each_chat[2:-1]))
                            conversation = []
                            

        src_tgt_dataset = tf.data.Dataset.zip((tf.data.Dataset.from_generator(lambda :src_data, tf.string), tf.data.Dataset.from_generator(lambda :tgt_data, tf.string)))

        #src_tgt_dataset = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(src_data), tf.data.Dataset.from_tensor_slices(tgt_data)))
        self.text_set = src_tgt_dataset

    def map_pyfunct(self, src, tgt):
        return [word for word in src.decode()],[word for word in tgt.decode()],1
    
    def map_split_func(self, src, tgt):
        a = tf.py_func(self.map_pyfunct, [src, tgt], [tf.string, tf.string, tf.int64])
        return a
    
    def _convert_to_tokens(self, buffer_size):
        #self.text_set = self.text_set.map(self.map_split_func).prefetch(buffer_size)
       # self.text_set = self.text_set.map(lambda src,tgt:(tf.string_split(src), tf.string_split(tgt))).prefetch(buffer_size)

        '''
        dataset = self.text_set
        dataset = dataset.batch(1)
        iter = dataset.make_one_shot_iterator()
        with tf.Session() as sess:
            a = sess.run(iter.get_next())
            print(a[0][0][0].decode())
            print(a[0][0][0].decode())
            print(a[1][0][0].decode())
            print(a[1][0][1].decode())
            print(a[1][0][2].decode())
            print(a[1][0][3].decode())
            print(a[1][0][4].decode())
            print(a[1][0][5].decode())
        '''

        self.id_set = self.text_set.map(lambda src, tgt:
                                        (tf.cast(self.vocab_table.lookup(src), tf.int32),
                                         tf.cast(self.vocab_table.lookup(tgt), tf.int32))
                                        ).prefetch(buffer_size)
        '''
        dataset = self.id_set
        dataset = dataset.batch(1)
        iter = dataset.make_initializable_iterator()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            sess.run(iter.initializer)
            a = sess.run(iter.get_next())
            print(a[0])
            print(a[1])
            a=1
        '''
        
  

def check_vocab(vocab_file):
    """Check to make sure vocab_file exists"""
    if tf.gfile.Exists(vocab_file):
        vocab_list = []
        with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as f:
            for word in f:
                vocab_list.append(word.strip())
    else:
        raise ValueError("The vocab_file does not exist. Please run the script to create it.")

    return len(vocab_list), vocab_list


class BatchedInput(namedtuple("BatchedInput",
                              ["initializer",
                               "source",
                               "target_input",
                               "target_output",
                               "source_sequence_length",
                               "target_sequence_length"])):
    pass
