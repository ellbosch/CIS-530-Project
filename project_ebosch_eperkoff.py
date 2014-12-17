#Main Project File for E. Margaret Perkoff and Elliot Boschwitz
from preprocessing_files import *
# from ngram_model import *
import xml.etree.ElementTree as ET
import collections
import itertools
import math
import os


train_labels_path = "/home1/c/cis530/project/train_labels.txt"
eperkoff_merged_dense_train_files_path = "/home1/e/eperkoff/CIS530/project/CIS-530-Project/merged_dense_train_files.txt"
eperkoff_merged_non_dense_train_files_path = "/home1/e/eperkoff/CIS530/project/CIS-530-Project/merged_non_dense_train_files.txt"
eperkoff_non_dense_trigram_model = "/home1/e/eperkoff/CIS530/project/CIS-530-Project/non_dense_trigram_model.srilm"
eperkoff_dense_trigram_model = "/home1/e/eperkoff/CIS530/project/CIS-530-Project/dense_trigram_model.srilm"
STOP_WORDS = []
W2V_VEC = {}

#Calculate the KL divergence threshold for articles in the training data

#takes in a file containing training labels of the following format: filename label leadoverlap and returns a tuple where the first value is a list of all the NON information dense files and the second is  alist of all the information dense files
def read_train_labels(file_path):
	not_dense_filenames = []
	dense_filenames = []
	open_train = open(file_path, 'r')
	for line in open_train:
		split = line.split(' ')
		if len(split) > 1:
			file_name = split[0]
			label = split[1]
			if label == '1':
				dense_filenames.append(file_name)
				print label + " dense"
			else:
				print label + " not dense"
				not_dense_filenames.append(file_name)
				
	open_train.close()
	return (not_dense_filenames, dense_filenames)

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
	return sorted(words_dict.keys(), key=lambda x: words_dict[x], reverse=True)[:2000]

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
		# w2v = open('/project/cis/nlp/tools/word2vec/vectors.txt', 'r')
		w2v = open('vectors.txt', 'r')

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
<<<<<<< HEAD
	#TODO: Elliot's code goes here
	return []
def calculate_kl_divergence(corpus_unigram, file_unigram):
	#TODO: Elliot's code goes here
	return []

def get_corpus_unigrams():
	#TODO: should give us the unigram for the corpus based on the normalized frequencies of the list of 2000 top words
	return []
=======
	unigram_vector = map_unigrams(xml_file, top_words)
	words_nonzero = [top_words[i] for i in range(len(top_words))
					 if unigram_vector[i] == 1]
	
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
>>>>>>> d6872e5f150419a32c9214e7fc5cfb168ef2bbe0

#gets the processed xml files corresponding to the non_dense and dense_files
def get_processed_file_lists(non_dense_files, dense_files):
	clean_non_dense_files = []
	clean_dense_files = []
	for non_dense in non_dense_files:
		clean_xml = eperkoff_processed_train_files_as_text_path + '/' + non_dense.replace('.txt', '_clean.txt')
		clean_non_dense_files.append(clean_xml)
	for dense in dense_files:
		clean_xml = eperkoff_processed_train_files_as_text_path + '/' + dense.replace('.txt', '_clean.txt')
		clean_dense_files.append(clean_xml)
	return (clean_non_dense_files, clean_dense_files)

#given a list of file names and a unigram for the corpus of files, this returns the following kl_divergence stats for the whole list: [min kl divergence value, max kl divergence value, average kl divergence value]
def calculate_kl_divergence_stats(file_names, corpus_unigram):
	kl_list = []
	sum = 0
	for fname in file_names:
		#TODO: figure out how to incorporate top_words & similarity matrix here
		file_unigram = map_expanded_unigrams(fname, top_words, similarity_matrix)
		kl = calculate_kl_divergence(corpus_unigram, file_unigram)
		kl_list.append(kl)
		sum += kl
	min_kl = min(kl_list)
	max_kl = max(kl_list)
	avg_kl = sum/(len(kl_list))
	return [min_kl, max_kl, avg_kl]

#given a list of file names, a representative unigram for the corpus of files, and a threshold for kl divergence, this returns two lists of the files from test_files: those that have a kl divergence lower than the threshold and those that have a kl divergence value higher than the threshold
def sort_out_of_range_kl_values(test_files, corpus_unigram, threshold):
	#list of files who have a kl value lower than the threshold	
	under_kl = []
	#list of files who have a kl value higher than the theshold
	over_kl = []
	for fname in test_files:
		#TODO: figure out how to incorporate top_words and similarity matrix here
		file_unigram = map_expanded_unigrams(fname, top_words, similarity_matrix)
		kl = calculate_kl_divergence(corpus_unigram, file_unigram)
		if (kl < threshold):
			under_kl.append(fname)
		else:
			over_kl.append(fname)
	return (under_kl, over_kl)

		
#THINGS TO RUN GO HERE#
'''
#ALL TRIGRAM MODELS MADE 
#generate lists of labels
label_lists = read_train_labels(train_labels_path)
print "Done reading the training labels"
#getting the xml files
actual_label_lists = get_processed_file_lists(label_lists[0], label_lists[1])
non_dense_files = actual_label_lists[0]
#print non_dense_files
dense_files = actual_label_lists[1]
#print dense_files

#preparing files to be sent to srilm to make the trigram model
process_files_for_srilm(dense_files, eperkoff_merged_dense_train_files_path)
print "Processed dense files for srilm"
generate_srilm_trigram_model(eperkoff_merged_dense_train_files_path, eperkoff_dense_trigram_model)
print "Made dense trigram model"
process_files_for_srilm(non_dense_files, eperkoff_merged_non_dense_train_files_path)
print "Processed non dense files for srilm"
generate_srilm_trigram_model(eperkoff_merged_non_dense_train_files_path, eperkoff_non_dense_trigram_model)
print "Made non dense trigram model"

'''
#generate the corpus unigram
corpus_unigram = get_corpus_unigrams()

#getting the stats on the kl divergence values
non_dense_stats = calculate_kl_divergence_stats(non_dense_files)
dense_stats = calculate_kl_divergence_stats(dense_files)

#set the threshold for kl divergence values here
threshold = 10

#get a list of all the processed test files' names
test_file_list = get_all_files(eperkoff_processed_test_data, [])
sorted_kl_value_lists = sort_out_of_range_kl_values(test_file_list, corpus_unigram, threshold)
	
