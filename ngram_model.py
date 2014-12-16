#Creates a trigram model for the corpus using the SRILM trigram format
import os
from preprocessing_files import *

eperkoff_SRILM_executable_path = "/home1/c/cis530/hw2/srilm/"

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
	
			
#takes a directory of files to be processed and an outputfile name to send the results to; merges all the files from directory into outputfile with all STOP instances replaced with \n
def process_files_for_srilm(directory, outputfile):
	write_file= open(outpfile, 'w')
	all_file_names = get_all_files(directory)
	for fname in all_file_names):
		file_string = clean_file_for_srilm(fname)
		write_file.write(file_string)
	write_file.close()

eperkoff_ngram_call = "ngram-count -unk -text "

#generates an srilm trigram model with Ney's absolute discounting of 0.75 and interpolation in MODEL_FILE based on the data in TEXT_FILE
def generate_srilm_trigram_model(TEXT_FILE, MODEL_FILE):
	call_string = 'cd '+ eperkoff_SRILM_executable_path + ' && ' +eperkoff_ngram_call + TEXT_FILE + " -lm " + MODEL_FILE + " -cdiscount 0.75 -interpolate"
	os.system(call_string)




#####RUN THIS######
