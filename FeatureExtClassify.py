'''Python script to process and extract inaugural speeches, extract and classify features'''

import os
import glob
import pandas as pd

# Referred to Lab 3 Part 1 processing inaugural speeches - NLP course

# Sort files once path to files to process are referenced
files = sorted(glob.glob(os.path.join('texts/', '*.csv' )))

def process_inaugural_speeches(files):
 # Define a corpus to process
 inaugural_speeches = {}

 for speech_file in files:
   filename = os.path.basename(speech_file)

   with open(speech_file, 'r', encoding="utf-8", errors="ignore") as file:
     speech_content = pd.read_csv(file, usecols=["speech", "party", "major_heading", "year"])
     inaugural_speeches[filename] = speech_content

     for i, row in speech_content.iterrows():
      # Define columns to process 
      rtn_speech = row["speech"]
      rtn_byline =  row["major_heading"]
      rtn_year =  row["year"]
      rtn_party =  row["party"]

     print(f"Political party: {rtn_party} \n-----\n  Year: {rtn_year} \n-----\n  Speech Byline: {rtn_byline}  \n-----\n  Speech: {rtn_speech}")
 
if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
process_inaugural_speeches(files) 