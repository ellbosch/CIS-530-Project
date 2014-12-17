#All early run files are put here
from project_ebosch_eperkoff import *


'''
#ALL OF THESE FILES HAVE BEEN MADE
#generate lists of labels
label_lists = read_train_labels(train_labels_path)
print "Done reading the training labels"
#getting the xml files
actual_label_lists = get_processed_file_lists(label_lists[0], label_lists[1])
non_dense_files = actual_label_lists[0]
dense_files = actual_label_lists[1]
df = open(dense_filenames, 'w')
for d in dense_files:
	df.write(d + '\n')
df.close()
print "Made dense file names file"
ndf = open(non_dense_filenames, 'w')
for nd in non_dense_files:
	ndf.write(nd + '\n')
ndf.close()
print "Made non dense file names file"

#ALL TRIGRAM MODELS MADE 
#preparing files to be sent to srilm to make the trigram model
process_files_for_srilm(dense_files, eperkoff_merged_dense_train_files_path)
print "Processed dense files for srilm"
generate_srilm_trigram_model(eperkoff_merged_dense_train_files_path, eperkoff_dense_trigram_model)
print "Made dense trigram model"
process_files_for_srilm(non_dense_files, eperkoff_merged_non_dense_train_files_path)
print "Processed non dense files for srilm"
generate_srilm_trigram_model(eperkoff_merged_non_dense_train_files_path, eperkoff_non_dense_trigram_model)
print "Made non dense trigram model"


top_words = extract_top_words(eperkoff_processed_files_path)
twf = open(top_words_file, 'w')
for tw in top_words:
	twf.write(tw + '\n')
twf.close()
print "Made top words file"


corpus_unigram = map_unigrams_corpus(eperkoff_processed_files_path, top_words)
cuf = open(corpus_unigram_file, 'w')
cuf.write(str(corpus_unigram))
cuf.close()

similarity_matrix = extract_similarity(top_words)
print similarity_matrix
smf = open(similarity_matrix_file, 'w')
smf.write(str(similarity_matrix))
smf.close()
print "Made similarity matrix"

'''
