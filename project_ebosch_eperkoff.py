#Main Project File for E. Margaret Perkoff and Elliot Boschwitz
from preprocessing_files import *
from ngram_model import *
from unigrams import *

import os


train_labels_path = "/home1/c/cis530/project/train_labels.txt"
eperkoff_merged_dense_train_files_path = "/home1/e/eperkoff/CIS530/project/CIS-530-Project/merged_dense_train_files.txt"
eperkoff_merged_non_dense_train_files_path = "/home1/e/eperkoff/CIS530/project/CIS-530-Project/merged_non_dense_train_files.txt"
eperkoff_non_dense_trigram_model = "/home1/e/eperkoff/CIS530/project/CIS-530-Project/non_dense_trigram_model.srilm"
eperkoff_dense_trigram_model = "/home1/e/eperkoff/CIS530/project/CIS-530-Project/dense_trigram_model.srilm"
dense_filenames = "/home1/e/eperkoff/CIS530/project/CIS-530-Project/dense_filenames.txt"
non_dense_filenames = "/home1/e/eperkoff/CIS530/project/CIS-530-Project/non_dense_filenames.txt"
top_words_file = "/home1/e/eperkoff/CIS530/project/CIS-530-Project/top_words.txt"
corpus_unigram_file = "/home1/e/eperkoff/CIS530/project/CIS-530-Project/corpus_unigram.txt"
similarity_matrix_file = "/home1/e/eperkoff/CIS530/project/CIS-530-Project/similarity_matrix.txt"
test_output_file = "/home1/e/eperkoff/CIS530/project/CIS-530-Project/test.txt"
eperkoff_dense_files_dir = "/home1/e/eperkoff/CIS530/project/CIS-530-Project/dense_files_dir"

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
			else:
				not_dense_filenames.append(file_name)
	open_train.close()
	return (not_dense_filenames, dense_filenames)


#gets the processed xml files corresponding to the non_dense and dense_files
def get_processed_file_lists(non_dense_files, dense_files):
	clean_non_dense_files = []
	clean_dense_files = []
	for non_dense in non_dense_files:
		clean_xml =eperkoff_processed_files_path + '/' + non_dense + '.xml'
		#print clean_xml
		clean_non_dense_files.append(clean_xml)
	for dense in dense_files:
		clean_xml = eperkoff_processed_files_path + '/' + dense + '.xml'
		clean_dense_files.append(clean_xml)
	return (clean_non_dense_files, clean_dense_files)

#given a list of file names, a unigram for the corpus of files, top_words for the corpus, and a similarity matrix for the corpus, this returns the following kl_divergence stats for the whole list: [min kl divergence value, max kl divergence value, average kl divergence value]
def calculate_kl_divergence_stats(file_names, corpus_unigram, top_words, similarity_matrix):
	kl_list = []
	sum = 0
	for fname in file_names:
		file_unigram = map_expanded_unigrams(fname, top_words, similarity_matrix)
		kl = calculate_kl_divergence(corpus_unigram, file_unigram)
		kl_list.append(kl)
		sum += kl
	min_kl = min(kl_list)
	max_kl = max(kl_list)
	avg_kl = sum/(len(kl_list))
	return [min_kl, max_kl, avg_kl]

#given a list of file names, a representative unigram for the corpus of files, top_words for the corpus, a similarity matrix for the corpus, and a threshold for kl divergence, this returns two lists of the files from test_files: those that have a kl divergence lower than the threshold and those that have a kl divergence value higher than the threshold
def sort_out_of_range_kl_values(test_files, corpus_unigram, top_words, similarity_matrix, threshold):
	#list of files who have a kl value lower than the threshold	
	under_kl = []
	#list of files who have a kl value higher than the theshold
	over_kl = []
	for fname in test_files:
		file_unigram = map_expanded_unigrams(fname, top_words, similarity_matrix)
		kl = calculate_kl_divergence(corpus_unigram, file_unigram)
		if (kl < threshold):
			under_kl.append(fname)
		else:
			over_kl.append(fname)
	print "Dense before trigram = " + str(len(under_kl))
	print "Nondense before trigram = " + str(len(over_kl))
	return (under_kl, over_kl)


def label_files(test_files, corpus_unigram, top_words, similarity_matrix, threshold):
	under_kl, over_kl = sort_out_of_range_kl_values(test_files, corpus_unigram, top_words, similarity_matrix, threshold)
	non_dense = over_kl
	dense = []
	for fname in under_kl:
		prob_non_dense = get_probability_of_article(fname, eperkoff_non_dense_trigram_model)
		prob_dense = get_probability_of_article(fname, eperkoff_dense_trigram_model)
		if prob_non_dense > prob_dense:
			non_dense.append(fname)
		else:
			dense.append(fname)
	print "Dense after trigram= " + str(len(dense))
	print "Nondense after trigram= " + str(len(non_dense))
	return (non_dense, dense)

def label_files_trigram_first(test_files, corpus_unigram, top_words, similarity_matrix, threshold):
	dense = []
	non_dense = []
	count_high_non_dense = 0
	count_high_dense = 0
	for fname in test_files:
		prob_non_dense = get_probability_of_article(fname, eperkoff_non_dense_trigram_model)
		prob_dense = get_probability_of_article(fname, eperkoff_dense_trigram_model)
		if prob_non_dense > prob_dense:
			non_dense.append(fname)
			count_high_non_dense +=1
		else:
			count_high_dense +=1
			file_unigram = map_expanded_unigrams(fname, top_words, similarity_matrix)
			kl = calculate_kl_divergence(corpus_unigram, file_unigram)
			if (kl < threshold):
				dense.append(fname)
			else:
				non_dense.append(fname)
	print "High probability of non-dense : " + str(count_high_non_dense)
	print "High probability of dense : " + str(count_high_dense)
	print "Number of dense files = " + str(len(dense))
	print "Number of non-dense files = " + str(len(non_dense))
	return (non_dense, dense)


def write_labels_to_file(non_dense_files, dense_files, output_file):
	of = open(output_file, 'w')
	for df in dense_files:
		fname = df.replace(eperkoff_processed_test_data + '/', '').replace('.xml', '')
		of.write(fname + ' 1 \n')
	for ndf in non_dense_files:
		fname = ndf.replace(eperkoff_processed_test_data + '/', '').replace('.xml', '')
		of.write(fname + ' -1 \n')
	of.close()



#THINGS TO RUN GO HERE#
df = open(dense_filenames, 'r')
dense_file_names = [line.replace('\n', '')  for line in df]
df.close()

ndf = open(non_dense_filenames, 'r')
non_dense_file_names = [line.replace('\n', '')  for line in ndf]
ndf.close()
#generate the corpus unigram
twf = open(top_words_file, 'r')
top_words = [line.replace('\n', '') for line in twf]
twf.close()

cuf = open(corpus_unigram_file, 'r')
split = cuf.read().split(', ')
corpus_unigram = [float(x.replace('[', '').replace(']', '')) for x in split]
cuf.close()

simfile = open(similarity_matrix_file, 'r')
simfileread = simfile.read()
simfile.close()
#takes out the outer brackets
simfileread2 = simfileread[1:-1]
list_of_dicts = re.findall('[\w\']+\:\ {[^{]+}', simfileread2)
similarity_matrix = {}
for d in list_of_dicts:
	key = re.findall('[\w\']+', d)[0]
	entries	= re.findall('[\w\']+\:\ [\w\.]+', d)
	temp_matrix = {}
	for e in entries:
		key2 = unicode(re.findall('\w+', e)[1], 'utf-8')
		value = float(re.findall('[0123456789]+\.[0123456789]+', e)[0])
		temp_matrix[key2] = value
	similarity_matrix[d] = temp_matrix
print "Read similarity matrix"
print "Read everything in from all files"



#dense_stats = calculate_kl_divergence_stats(dense_file_names, corpus_unigram, top_words, similarity_matrix)
#print "Dense stats " + str(dense_stats) # [9270.203727004502, 9665.161967964274, 9348.089721787068]
#Dense stats With 500: [21.299993567723927, 416.26011623558463, 131.68439360199238]
#Dense stats with 100:  [0.0, 152.96692451543237, 54.131074485557946]
#Dense stats with 50: [0.0, 99.88396207209955, 34.70559224016129]
#With 10 words:  [0.0, 42.51331714455911, 11.592345762291863]
# [0.0, 46.387984384514326, 11.384570932477654]




#non_dense_stats = calculate_kl_divergence_stats(non_dense_file_names, corpus_unigram, top_words, similarity_matrix)
#print "Non-dense stats " + str(non_dense_stats) #[9245.28769357885, 9538.31354291483, 9319.187375587724]
# With 500: [5.812127543091245, 242.92532069780748, 93.13380797903638]
#Non-Dense stats with 100:  [0.0, 128.06780000980535, 36.441087271071936]
#Non Dense stats with 50: [0.0, 91.73330048858023, 24.105637764702532]
#With 10 words: [0.0, 36.68633624476288, 7.885628591399675]
#[0.0, 35.24578247088941, 6.9662633286377185]



#set the threshold for kl divergence values here
threshold = 11.384570932477654

test_files = get_all_files(eperkoff_processed_test_data, [])
predicted_labels = label_files_trigram_first(test_files, corpus_unigram, top_words, similarity_matrix, threshold)

print "Generated predicted_labels"
predicted_non_dense = predicted_labels[0]
predicted_dense = predicted_labels[1]
write_labels_to_file(predicted_non_dense, predicted_dense, test_output_file)
print "Wrote all labels to file"

