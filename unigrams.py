#All of the functions to be used to create unigrams
import xml.etree.ElementTree as ET
import collections
import itertools
import math
import os

STOP_WORDS = []
W2V_VEC = {}

# returns tuple of a dict of words with their frequency and a count of all words
def extract_frequencies(xml_directory):
	# lazy instantiation for STOP_WORDS
	global STOP_WORDS
	if not STOP_WORDS:
		f_stopwords = open('stopwords.txt', 'r')
		STOP_WORDS = [word.rstrip('\n') for word in f_stopwords]
		f_stopwords.close()

	words_dict = collections.defaultdict(int)
	files = os.listdir(xml_directory)
	total_count = 0

	for f in files:
		xml = open(xml_directory + '/' + f, 'r')
		tree = ET.parse(xml)
		root = tree.getroot()

		for token in root.iter('token'):
			total_count += 1
			word = token.findall('word')[0].text.lower()

			# only add word if it isn't a stop word
			if word not in STOP_WORDS:
				words_dict[word] += 1

		xml.close()

	return words_dict, total_count

#extracts top words of xml files from directory
def extract_top_words(xml_dir):
	words_dict, _ = extract_frequencies(xml_dir)
	return sorted(words_dict.keys(), key=lambda x: words_dict[x], reverse=True)[:10]

#creates unigram model over corpus
def map_unigrams_corpus(xml_dir, top_words):
	words_dict, total_count = extract_frequencies(xml_dir)
	freq = {word: float(freq) / total_count for word, freq in words_dict.items()}
	return [freq[word] for word in top_words]


#creates vector in feature space of top_words, to be used for map_expanded_unigrams()
def map_unigrams(xml_filename, top_words):
	f = open(xml_filename, 'r')

	all_words = []
	output_list = []

	tree = ET.parse(f)
	root = tree.getroot()

	for t in root.iter('token'):
		word = t.findall('word')[0].text.lower()
		all_words.append(word)
	f.close()

	for word in top_words:
		if word in all_words:
			output_list.append(1)
		else:
			output_list.append(0)

	return output_list

#creates similarity matrix
def extract_similarity(top_words):
	# lazily instantiate word2vec vectors
	global W2V_VEC
	if not W2V_VEC:
		'''  ************************************
		CHANGE THE PATH BELOW TO ENIAC BEFORE SUBMISSION!!
		****************************************'''
	
		w2v = open('/project/cis/nlp/tools/word2vec/vectors.txt', 'r')
		#w2v = open('vectors.txt', 'r')

		vectors = itertools.islice(w2v, 1, None)
		for line in vectors:
			tokens = line.split()
			word = tokens[0]
			W2V_VEC[word] = [float(t) for t in tokens[1:]]

	output_dict = {}
	words_list = [word for word in top_words if word in W2V_VEC.keys()]
	for w1 in words_list:
		value_dict = {}

		other_words = [word for word in words_list if word != w1]
		for w2 in other_words:
			w1_vectors = W2V_VEC[w1]
			w2_vectors = W2V_VEC[w2]
			n = min(len(w1_vectors), len(w2_vectors))

			numerator, x_sum_sq, y_sum_sq = 0, 0, 0
			for i in range(n):
				x, y = w1_vectors[i], w2_vectors[i]

				numerator += x * y
				x_sum_sq += x ** 2
				y_sum_sq += y ** 2

			value_dict[w2] = numerator / math.sqrt(x_sum_sq * y_sum_sq)

		output_dict[w1] = value_dict

	return output_dict

def map_expanded_unigrams(xml_file, top_words, similarity_matrix):
	unigram_vector = map_unigrams(xml_file, top_words)
	#words_nonzero = [top_words[i] for i in range(len(top_words))
					 #if unigram_vector[i] == 1]
	
	for i in range(len(top_words)):
		if unigram_vector[i] == 0 and top_words[i] in similarity_matrix.keys():
			word = top_words[i]
			unigram_vector[i] = max(similarity_matrix[word].values())

	return unigram_vector

def calculate_kl_divergence(corpus_unigram, file_unigram):
	kl_sum = 0.0

	for i in range(len(corpus_unigram)):
		q = corpus_unigram[i]
		p = file_unigram[i]

		if p != 0:
			kl_sum += p * math.log(float(p) / q)

	return kl_sum
