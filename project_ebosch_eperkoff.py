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
	for fname in file_names:
		#TODO: figure out how to incorporate top_words & similarity matrix here
		map_expanded_unigrams(fname, top_words, similarity_matrix)
		
		
	
	

