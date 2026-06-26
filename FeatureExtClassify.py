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
        "rtn_speech": str(row["speech"].strip().lower()),
        "rtn_byline":str(row["major_heading"].strip().lower()),
        "rtn_year":int(row["year"].strip()),
        "rtn_party":str(row["party"].strip().lower()),
        "rtn_speech_class": str(row["speech_class"].strip().lower()),
        "rtn_speaker": str(row["speaker"].strip().lower())
      }

      print(f"Political party: {columns['rtn_party']} \n-----\n  Year: {columns['rtn_year']} \n-----\n  Speech Byline: {columns['rtn_byline']}  \n-----\n  Speech: {columns['rtn_speech']}")
 
      # begin processing and updating csv
      val_party = columns["rtn_party"]
      val_speaker = columns["rtn_speaker"]
      val_speech_class = columns["rtn_speech_class"]

      try:
          print(f"Returning party text value on row {i}: '{val_party}'")

          # 1. condition for update to party name
          if val_party == 'labour (co-op)':

            print(f"[MATCH] Found party text on row {i}: '{val_party}'")
            columns["rtn_party"] = val_party.replace('labour (co-op)', 'Labour')

            # Update dataFrame and save csv to disk
            df.at[i, "party"] = columns["rtn_party"]
            
            print(f"Replaced party column values: {val_party} with {columns['rtn_party']}")

          # 2. remove the speaker value and then remove rows where party name is not one of the big 4 (Labour, Liberal, Conservative, etc. )
          allowed_parties = ['labour', 'conservative', 'liberal', 'unionist']
          # first check if the current row's party contains ANY of your allowed big 4 keywords
          is_valid_party = any(party in val_party for party in allowed_parties) 
            
          if "speaker" in val_party or "speaker" in val_speaker:
              print(f"[DROP] Row {i}: Removing Speaker record.")
              df.drop(i, inplace=True)
              continue  # Skip to the next row immediately
        
          if not is_valid_party or val_party in ['', 'nan', ' ']:
              print(f"[DROP] Row {i}: Removing non-standard party text: '{columns['rtn_party']}'")
              df.drop(i, inplace=True)
              continue  # Skip to the next row immediately  
          
              # Update dataFrame and save csv to disk
              df.at[i, "speaker"] = columns["rtn_speaker"]
              df.at[i, "party"] = columns["rtn_party"]

          # 3. remove rows where ‘speech class’ column is not ‘Speech’
          # Drop missing NaN entries first so string length calculations don't crash
          df = df.dropna(subset=['speech', 'speech class'])
          df['speech class'] = df['speech class'].astype(str).str.strip()
          print(f"Row count after Rule 3 (Speech class filter): {len(df)}")
          df.at[i, "speech_class"] = columns["rtn_speech_class"]

          # 4. remove rows where ‘speech’ column has < 1000 chars long
          # Keep only rows where string character length is 1000 or greater
          df = df[df['speech'].astype(str).str.len() >= 1000]
          print(f"Row count after Rule 4 (1000+ character filter): {len(df)}")
          df.at[i, "speech"] = columns["rtn_speech"]
          
          # save to file after updates are made
          df.to_csv(csv_to_process, index=False, encoding="utf-8")

          # return the shape of the DataFrame 
          num_rows = df.shape[0]
          num_cols = df.shape[1]

          print(f"The data currently has {num_rows} rows and {num_cols} columns")

      except ValueError as e:
          print(f"Error returned when replacing column values for {e}")

      #Print confirmartion after the loop finishes processing all rows
      print(f"\n[DONE] Processing complete.")

def vectorise_speeches(df):
   '''Vectorise the speeches using TfidfVectorizer from scikit-learn''' 
   # Referred to: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html


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