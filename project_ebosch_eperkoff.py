#Main Project File for E. Margaret Perkoff and Elliot Boschwitz
from preprocessing_files import *
from ngram_model import *

train_labels_path = "/home1/c/cis530/project/train_labels"

#Calculate the KL divergence threshold for articles in the training data

#takes in a file containing training labels of the following format: filename label leadoverlap and returns a tuple where the first value is a list of all the NON information dense files and the second is  alist of all the information dense files
def read_train_labels(file_path):
	not_dense_filenames = []
	dense_filenames = []
	open_train = open(file_path, 'r')
	for line in open_train:
		split = line.split(' ')
		if len(split > 1):
			file_name = split[0]
			label = split[1]
			if label == 0:
				not_dense_filenames.append(file_name)
			else:
				dense_filenames.append(file_name)
	open_train.close()
	return (not_dense_filenames, dense_filenames)

def map_expanded_unigrams(xml_file, top_words, similarity_matrix):
	#TODO: Elliot's code goes here

def calculate_kl_divergence(corpus_unigram, file_unigram):
	#TODO: Elliot's code goes here

def get_corpus_unigrams():
	#TODO: should give us the unigram for the corpus based on the normalized frequencies of the list of 2000 top words

#gets the processed xml files corresponding to the non_dense and dense_files
def get_processed_file_lists(non_dense_files, dense_files):
	clean_non_dense_files = []
	clean_dense_files = []
	for non_dense in non_dense_files:
		clean_xml = non_dense + '.xml'
		clean_non_dense_files.append(clean_xml)
	for dense in dense_files:
		clean_xml = dense + '.xml'
		clean_dense_files.append(clean_xml)
	return (clean_non_dense_files, clean_dense_files)

#given a list of file names and a unigram for the corpus of files, this returns the following kl_divergence stats for the whole list: [min kl divergence value, max kl divergence value, average kl divergence value]
def calculate_kl_divergence_stats(file_names, corpus_unigram):
	kl_list = []
	sum = 0
	for fname in file_names:
		#TODO: figure out how to incorporate top_words & similarity matrix here
		file_unigram = map_expanded_unigrams(fname, top_words, similarity_matrix)
		kl = calculate(kl_divergence(corpus_unigram, file_unigram)
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
#generate lists of labels
label_lists = read_train_labels(train_labels_path)
#getting the xml files
actual_label_lists = get_processed_file_lists(label_lists[0], label_lists[1])
non_dense_files = actual_label_lists[0]
dense_files = actual_label_lists[1]

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
		
		
	
	

