#Creates a trigram model for the corpus using the SRILM trigram format
import os
import re
import subprocess
from preprocessing_files import *


eperkoff_non_dense_trigram_model = "/home1/e/eperkoff/CIS530/project/CIS-530-Project/non_dense_trigram_model.srilm"
eperkoff_SRILM_executable_path = "/home1/c/cis530/hw2/srilm/"
eperkoff_test_thing = "/home1/e/eperkoff/CIS530/project/CIS-530-Project/temp_file.txt"
eperkoff_test_thing2 = "/home1/e/eperkoff/CIS530/project/CIS-530-Project/temp_file2.txt"
eperkoff_ppl_output = "/home1/e/eperkoff/CIS530/project/CIS-530-Project/ppl_output.txt"
#removes all of the STOP instances in a file and replaces them with a newline, returns a string of all the words in the file
def clean_file_for_srilm(inputfile):
	f = open(inputfile, 'r')
	file_string = ""
	for line in f:
		#replaces all STOP instances in this line with a newline character
		new_line = re.subn('STOP', '\n', line)[0]
		file_string += new_line
	f.close()
	return file_string
	
			
#takes a list of files to be processed and an outputfile name to send the results to; merges all the files from directory into outputfile with all STOP instances replaced with \n
def process_files_for_srilm(all_file_names, outputfile):
	write_file= open(outputfile, 'w')
	for fname in all_file_names:
		file_string = clean_file_for_srilm(fname)
		write_file.write(file_string)
	write_file.close()

eperkoff_ngram_call = "ngram-count -unk -text "

#generates an srilm trigram model with Ney's absolute discounting of 0.75 and interpolation in MODEL_FILE based on the data in TEXT_FILE
def generate_srilm_trigram_model(TEXT_FILE, MODEL_FILE):
	call_string = 'cd '+ eperkoff_SRILM_executable_path + ' && ' +eperkoff_ngram_call + TEXT_FILE + " -lm " + MODEL_FILE + " -cdiscount 0.75 -interpolate"
	os.system(call_string)

#gets the log probability of an article given an srilm trigram model 
def get_probability_of_article(xml_file, model_file):
	process_file(xml_file, eperkoff_test_thing)
	process_files_for_srilm([eperkoff_test_thing], eperkoff_test_thing2)
	call_string = call_string = 'cd '+ eperkoff_SRILM_executable_path + ' && ' +"ngram -unk -lm " + model_file+" -ppl "  + eperkoff_test_thing2
	temp_string = subprocess.check_output(call_string, shell=True)
	split = temp_string.split(' ')
	temp_index = split.index('logprob=')
	log_prob = float(split[temp_index +1])
	return log_prob
	

#process_file(eperkoff_processed_test_data+"/1996_09_26_0879509.txt.xml", eperkoff_test_thing)
#process_files_for_srilm([eperkoff_test_thing], eperkoff_test_thing2)
#print get_probability_of_article(eperkoff_processed_test_data+"/1996_09_26_0879509.txt.xml", #eperkoff_non_dense_trigram_model)
	




