'''Python script to process and extract inaugural speeches, extract and classify features'''

import os
import glob
import pandas as pd

# Referred to Lab 3 Part 1 processing inaugural speeches - NLP course

def process_inaugural_speeches(df):
 # Define a corpus to process
 inaugural_speeches = {}

 for i, row in df.iterrows():
      
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
# Store the file in a DataFeame, then sort files once path to file to processis referenced
csv_file = sorted(glob.glob(os.path.join('texts/', 'hansard10000.csv' )))
csv_to_process = csv_file[0]
try:
  with open(csv_to_process, 'r', encoding="utf-8", errors="ignore") as file:
   df = pd.read_csv(file, usecols=["speech", "party", "major_heading", "year"])
   process_inaugural_speeches(df)
except ValueError as e:
  print(f"Unable to process {csv_to_process} {e}")