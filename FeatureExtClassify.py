'''Python script to process and extract inaugural speeches, extract and classify features'''

import os
import glob

# Referred to Lab 3 Part 1 processing inaugural speeches - NLP course

# Sort files once path to files to process are referenced
files = sorted(glob.glob(os.path.join('texts/', '*.csv' )))

def process_inaugural_speeches(files):
 # Define a corpus to process
 inaugural_speeches = {}

 for speech_file in files:
   filename = os.path.basename(speech_file)

   with open(speech_file, 'r', encoding="utf-8", errors="ignore") as file:
     speech_content = file.read()
     inaugural_speeches[filename] = speech_content

     print(speech_content)
 
if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
process_inaugural_speeches(files) 