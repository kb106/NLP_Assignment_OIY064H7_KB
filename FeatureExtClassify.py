'''Python script to process and extract inaugural speeches, extract and classify features'''

import os
import glob
import pandas as pd

# Referred to Lab 3 Part 1 processing inaugural speeches - NLP course

def process_inaugural_speeches(df):
 # Define a corpus to process
 inaugural_speeches = {}
 colval_count = 0

 for i, row in df.iterrows():
      
      # Define columns to process
      columns = {
        "rtn_speech": str(row["speech"]),
        "rtn_byline":str(row["major_heading"]),
        "rtn_year":row["year"],
        "rtn_party":str(row["party"])
      }

      # print(f"Political party: {columns['rtn_party']} \n-----\n  Year: {columns['rtn_year']} \n-----\n  Speech Byline: {columns['rtn_byline']}  \n-----\n  Speech: {columns['rtn_speech']}")
 
      # begin processing and updating csv
      val = columns["rtn_party"]

      try:
          # Updates might be case-sensitive so set all values in lower case first 
          val_lowercase = val.lower()
          print(f"Returning party text value on row {i}: '{val_lowercase}'")

          if val_lowercase == 'labour (co-op)':

            print(f"[MATCH] Found party text on row {i}: '{val_lowercase}'")

            print(f"Column rtn_year and value: {val_lowercase}")

            columns["rtn_party"] = val_lowercase.replace('labour (co-op)', 'Labour')

            df.at[i, "party"] = columns["rtn_party"]
            df.to_csv(csv_to_process, index=False, encoding="utf-8")
            
            print(f"Replaced party column values: {val_lowercase} with {columns['rtn_party']}")

            colval_count += 1

      except ValueError as e:
          print(f"Error returned when replacing column values for {e}")

      # Print the final complete tally after the loop finishes processing all rows
      #print(f"\n[DONE] Processing complete. Total rows updated: {colval_count}")

if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
# Store the file in a DataFeame, then sort files once path to file to processis referenced
csv_file = sorted(glob.glob(os.path.join('texts/', 'hansard10000.csv' )))
csv_to_process = csv_file[0]
try:
  with open(csv_to_process, 'r', encoding="utf-8", errors="ignore") as file:
   df = pd.read_csv(file)
   process_inaugural_speeches(df)
except ValueError as e:
  print(f"Unable to process {csv_to_process} {e}")