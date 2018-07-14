from __future__ import absolute_import, division, print_function

# Import TensorFlow >= 1.9 and enable eager execution
import tensorflow as tf

tf.enable_eager_execution()

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow.contrib.eager as tfe
import unicodedata
import re
import numpy as np
import os
import time
import jieba

num_examples = 1000
BATCH_SIZE = 64
embedding_dim = 256
units = 1024
EPOCHS = 1

print(tf.__version__)
# Download the file
# path_to_zip = tf.keras.utils.get_file(
#	'spa-eng.zip', origin='http://download.tensorflow.org/data/spa-eng.zip',
#	extract=True)

# path_to_file = os.path.dirname(path_to_zip) + "/spa-eng/spa.txt"
path_to_file = '/home/tizen/share/charmpy/tf/cmn-eng/a.txt'


# Converts the unicode file to ascii
def unicode_to_ascii(s):
	return ''.join(c for c in unicodedata.normalize('NFD', s)
	               if unicodedata.category(c) != 'Mn')


def preprocess_sentence_for_eng(w):
	w = unicode_to_ascii(w.lower().strip())
	
	# creating a space between a word and the punctuation following it
	# eg: "he is a boy." => "he is a boy ."
	# Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
	w = re.sub(r"[(.,!?\"':;)]【（。，！？、“：；）】", r" \1 ", w)
	w = re.sub(r'[" "]+', " ", w)
	
	# replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
	w = re.sub(r"[^a-zA-Z?.!,]+", " ", w)
	
	w = w.rstrip().strip()
	
	# adding a start and an end token to the sentence
	# so that the model know when to start and stop predicting.
	w = '<start> ' + w + ' <end>'
	return w


def preprocess_sentence_for_chi(w):
	# creating a space between a word and the punctuation following it
	# eg: "he is a boy." => "he is a boy ."
	# Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
	w = re.sub(r"[(.,!?\"':;)]【（。，！？、“：；）】", r" \1 ", w)
	
	w = ' '.join(jieba.cut(w, cut_all=False))
	w = re.sub(r'[" "]+', " ", w)
	
	w = w.rstrip().strip()
	
	# adding a start and an end token to the sentence
	# so that the model know when to start and stop predicting.
	w = '<start> ' + w + ' <end>'
	return w


# 1. Remove the accents
# 2. Clean the sentences
# 3. Return word pairs in the format: [ENGLISH, SPANISH]
def create_dataset(path, num_examples):
	lines = open(path, encoding='UTF-8').read().strip().split('\n')
	word_pairs = [[preprocess_sentence_for_eng(l.split('\t')[0]), preprocess_sentence_for_chi(l.split('\t')[1])] for l
	              in lines[:num_examples]]
	
	return word_pairs


# This class creates a word -> index mapping (e.g,. "dad" -> 5) and vice-versa
# (e.g., 5 -> "dad") for each language,
class LanguageIndex():
	def __init__(self, lang):
		self.lang = lang
		self.word2idx = {}
		self.idx2word = {}
		self.vocab = set()
		
		self.create_index()
	
	def create_index(self):
		for phrase in self.lang:
			self.vocab.update(phrase.split(' '))
		
		self.vocab = sorted(self.vocab)
		
		self.word2idx['<pad>'] = 0
		for index, word in enumerate(self.vocab):
			self.word2idx[word] = index + 1
		
		for word, index in self.word2idx.items():
			self.idx2word[index] = word


def max_length(tensor):
	return max(len(t) for t in tensor)


def load_dataset(path, num_examples):
	# creating cleaned input, output pairs
	pairs = create_dataset(path, num_examples)
	
	# index language using the class defined above
	inp_lang = LanguageIndex(sp for en, sp in pairs)
	targ_lang = LanguageIndex(en for en, sp in pairs)
	
	# Vectorize the input and target languages
	
	# Spanish sentences
	input_tensor = [[inp_lang.word2idx[s] for s in sp.split(' ')] for en, sp in pairs]
	
	# English sentences
	target_tensor = [[targ_lang.word2idx[s] for s in en.split(' ')] for en, sp in pairs]
	
	# Calculate max_length of input and output tensor
	# Here, we'll set those to the longest sentence in the dataset
	max_length_inp, max_length_tar = max_length(input_tensor), max_length(target_tensor)
	
	# Padding the input and output tensor to the maximum length
	input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor,
	                                                             maxlen=max_length_inp,
	                                                             padding='post')
	
	target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor,
	                                                              maxlen=max_length_tar,
	                                                              padding='post')
	
	return input_tensor, target_tensor, inp_lang, targ_lang, max_length_inp, max_length_tar


# Try experimenting with the size of that dataset

input_tensor, target_tensor, inp_lang, targ_lang, max_length_inp, max_length_targ = load_dataset(path_to_file,
                                                                                                 num_examples)

# Creating training and validation sets using an 80-20 split
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor,
                                                                                                target_tensor,
                                                                                                test_size=0.2)

# Show length
len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val)
BUFFER_SIZE = len(input_tensor_train)

vocab_inp_size = len(inp_lang.word2idx)
vocab_tar_size = len(targ_lang.word2idx)

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)

iter = tfe.Iterator(dataset)
a = iter.next()
print(a)
dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(BATCH_SIZE))


def gru(units):
	# If you have a GPU, we recommend using CuDNNGRU(provides a 3x speedup than GRU)
	# the code automatically does that.
	if tf.test.is_gpu_available():
		return tf.keras.layers.CuDNNGRU(units,
		                                return_sequences=True,
		                                return_state=True,
		                                recurrent_initializer='glorot_uniform')
	else:
		return tf.keras.layers.GRU(units,
		                           return_sequences=True,
		                           return_state=True,
		                           recurrent_activation='sigmoid',
		                           recurrent_initializer='glorot_uniform')


class Encoder(tf.keras.Model):
	def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
		super(Encoder, self).__init__()
		self.batch_sz = batch_sz
		self.enc_units = enc_units
		self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
		self.gru = gru(self.enc_units)
	
	def call(self, x, hidden):
		x = self.embedding(x)
		output, state = self.gru(x, initial_state=hidden)
		return output, state
	
	def initialize_hidden_state(self):
		return tf.zeros((self.batch_sz, self.enc_units))


class Decoder(tf.keras.Model):
	def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
		super(Decoder, self).__init__()
		self.batch_sz = batch_sz
		self.dec_units = dec_units
		self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
		self.gru = gru(self.dec_units)
		self.fc = tf.keras.layers.Dense(vocab_size)
		
		# used for attention
		self.W1 = tf.keras.layers.Dense(self.dec_units)
		self.W2 = tf.keras.layers.Dense(self.dec_units)
		self.V = tf.keras.layers.Dense(1)
	
	def call(self, x, hidden, enc_output):
		# enc_output shape == (batch_size, max_length, hidden_size)
		
		# hidden shape == (batch_size, hidden size)
		# hidden_with_time_axis shape == (batch_size, 1, hidden size)
		# we are doing this to perform addition to calculate the score
		hidden_with_time_axis = tf.expand_dims(hidden, 1)
		
		# score shape == (batch_size, max_length, hidden_size)
		score = tf.nn.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis))
		
		# attention_weights shape == (batch_size, max_length, 1)
		# we get 1 at the last axis because we are applying score to self.V
		attention_weights = tf.nn.softmax(self.V(score), axis=1)
		
		# context_vector shape after sum == (batch_size, hidden_size)
		context_vector = attention_weights * enc_output
		context_vector = tf.reduce_sum(context_vector, axis=1)
		
		# x shape after passing through embedding == (batch_size, 1, embedding_dim)
		x = self.embedding(x)
		
		# x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
		x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
		
		# passing the concatenated vector to the GRU
		output, state = self.gru(x)
		
		# output shape == (batch_size * max_length, hidden_size)
		output = tf.reshape(output, (-1, output.shape[2]))
		
		# output shape == (batch_size * max_length, vocab)
		x = self.fc(output)
		
		return x, state, attention_weights
	
	def initialize_hidden_state(self):
		return tf.zeros((self.batch_sz, self.dec_units))


encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

optimizer = tf.train.AdamOptimizer()


def loss_function(real, pred):
	mask = 1 - np.equal(real, 0)
	loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
	return tf.reduce_mean(loss_)




for epoch in range(EPOCHS):
	start = time.time()
	
	hidden = encoder.initialize_hidden_state()
	total_loss = 0
	
	for (batch, (inp, targ)) in enumerate(dataset):
		loss = 0
		
		with tf.GradientTape() as tape:
			enc_output, enc_hidden = encoder(inp, hidden)
			
			dec_hidden = enc_hidden
			
			dec_input = tf.expand_dims([targ_lang.word2idx['<start>']] * BATCH_SIZE, 1)
			
			# Teacher forcing - feeding the target as the next input
			for t in range(1, targ.shape[1]):
				# passing enc_output to the decoder
				predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
				
				loss += loss_function(targ[:, t], predictions)
				
				# using teacher forcing
				dec_input = tf.expand_dims(targ[:, t], 1)
		
		total_loss += (loss / int(targ.shape[1]))
		
		variables = encoder.variables + decoder.variables
		
		gradients = tape.gradient(loss, variables)
		
		optimizer.apply_gradients(zip(gradients, variables), tf.train.get_or_create_global_step())
		
		if batch % 100 == 0:
			print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
			                                             batch,
			                                             loss.numpy() / int(targ.shape[1])))
	
	print('Epoch {} Loss {:.4f}'.format(epoch + 1,
	                                    total_loss / len(input_tensor)))
	print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


def evaluate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ):
	attention_plot = np.zeros((max_length_targ, max_length_inp))
	
	sentence = preprocess_sentence_for_chi(sentence)
	
	inputs = [inp_lang.word2idx[i] for i in sentence.split(' ')]
	inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')
	inputs = tf.convert_to_tensor(inputs)
	
	result = ''
	
	hidden = [tf.zeros((1, units))]
	enc_out, enc_hidden = encoder(inputs, hidden)
	
	dec_hidden = enc_hidden
	dec_input = tf.expand_dims([targ_lang.word2idx['<start>']], 0)
	
	for t in range(max_length_targ):
		predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)
		
		# storing the attention weigths to plot later on
		attention_weights = tf.reshape(attention_weights, (-1,))
		attention_plot[t] = attention_weights.numpy()
		
		predicted_id = tf.multinomial(tf.exp(predictions), num_samples=1)[0][0].numpy()
		
		result += targ_lang.idx2word[predicted_id] + ' '
		
		if targ_lang.idx2word[predicted_id] == '<end>':
			return result, sentence, attention_plot
		
		# the predicted ID is fed back into the model
		dec_input = tf.expand_dims([predicted_id], 0)
	
	return result, sentence, attention_plot


# function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence):
	fig = plt.figure(figsize=(10, 10))
	ax = fig.add_subplot(1, 1, 1)
	ax.matshow(attention, cmap='viridis')
	
	fontdict = {'fontsize': 14}
	
	ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
	ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)
	
	plt.show()


def translate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ):
	result, sentence, attention_plot = evaluate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp,
	                                            max_length_targ)
	
	print('Input: {}'.format(sentence))
	print('Predicted translation: {}'.format(result))
	
	attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
	plot_attention(attention_plot, sentence.split(' '), result.split(' '))


translate('這是什麼啊？', encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
translate('什麼鳥兒？', encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)













































