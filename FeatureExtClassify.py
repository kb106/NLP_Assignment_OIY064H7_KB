'''Python script to process and extract inaugural speeches, extract and classify features'''

import os
import glob
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score
import numpy as np

# Referred to Lab 3 Part 1 processing inaugural speeches - NLP course

def process_inaugural_speeches(df):
  # Define a corpus to process
  inaugural_speeches = {}
  colval_count = 0

  # Define columns to process
  df.columns = df.columns.str.lower().str.strip() 

  # print(f"Political party: {columns['rtn_party']} \n-----\n  Year: {columns['rtn_year']} \n-----\n  Speech Byline: {columns['rtn_byline']}  \n-----\n  Speech: {columns['rtn_speech']}")
 
  # begin processing and updating csv
  for i, row in df.iterrows():
    try:
          val_party = str(row.get("party","")).lower().strip()
          print(f"Returning party text value on row {i}: '{val_party}'")

          # 1. condition for update to party name
          if val_party == 'labour (co-op)':

            print(f"[MATCH] Found party text on row {i}: '{val_party}'")
            # Update dataFrame and save csv to disk
            df.at[i, "party"] = "Labour"

    except ValueError as e:
            print(f"Error updating text on row {i} as {e}")

    continue  # Skip to the next row immediately

  # Process remaining updates to rows and columns, using below to set DataFrame to drop rows that comply with clean-up rules
  df = df.dropna(subset=['speech','speech_class','party','speakername'])
  # Define cols to process
  df["speech_class"] = df["speech_class"].astype(str).str.strip().str.lower()
  df["party"] = df["party"].astype(str).str.strip().str.lower()
  df["speakername"] = df["speakername"].astype(str).str.strip().str.lower()

  # 2. Remove any value outside the four most common parties 
  print(f"[DROP] Row {i}: Removing non-standard party from column: '{df["party"]}'")
  fourparties = df["party"].value_counts().head(4).index.tolist()
  print(f"The 4 most common parties found and kept are: {fourparties}")
  # Remove if not in top 4 parties
  df = df[df['party'].isin(fourparties)]

  # Remove the 'Speaker' value
  df = df[df['party'].str.lower() != 'speaker']
  df = df[df['speakername'] != 'speaker']

  # 3. remove rows where ‘speech class’ column is not ‘Speech’
  # Drop missing NaN entries first so string length calculations don't crash
  df = df[df['speech_class'] == 'speech']
  print(f"Row count after Rule 3 (Speech class filter): {len(df)}")

  # 4. remove rows where ‘speech’ column has < 1000 chars long
  # Keep only rows where string character length is 1000 or greater
  df = df[df['speech'].astype(str).str.len() >= 1000]
  print(f"Row count after Rule 4 (1000+ character filter): {len(df)}")
          
  # save to file after updates are made
  df.to_csv(csv_to_process, index=False, encoding="utf-8")

  # return the shape of the DataFrame 
  num_rows = df.shape[0]
  num_cols = df.shape[1]
  print(f"The data currently has {num_rows} rows and {num_cols} columns")

  #Print confirmartion after the loop finishes processing all rows
  print(f"\n[DONE] Processing complete.")

def vectorise_speeches(df):
   '''Vectorise the speeches using TfidfVectorizer from scikit-learn - 
   Label to predict is party classifier (i.e. to identify which political party likely made the speech) ''' 
   # Referred to: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
   # Call vectorizer using sklearn import for Tfidf, transform DataFrame text and print names

   # Accept defaults for Tfidf except stopwords and max_features
   stopwords = 'english'
   max_features = 3000
   # Sampling threshold
   random_seed = 26
   # Use RandomForests with estimator set
   n_estimators = 300  

   # Cols ot process cleaned into a list and set data type explicitly
   X_speechtext = df["speech"].astype(str).tolist()
   Y_partylabels  = df["party"].astype(str).tolist()

   # 1. Use AML to extract feature set 
   vectorizer = TfidfVectorizer(stop_words=stopwords, max_features=max_features)
   X_features = vectorizer.fit_transform(X_speechtext)
   print(f"Extraction vector dimensions or shape is: {X_features.shape}")

   # 2. Define training and test sets (split them from data)
   X_trainset, X_testset, Y_trainset, Y_testset  = train_test_split(
      X_features,
      Y_partylabels,
      test_size=0.2,
      random_state=random_seed,
      stratify=Y_partylabels
   )
   
   # TRAINING: For trinaing use SVM/ linear kernel classifiers on the training set
   rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_seed)
   rf_model.fit(X_trainset, Y_trainset)

   # TESTING: Print macro-average-f1-score and classification report for each classifier on the test set 
   rf_prediction = rf_model.predict(X_testset)
   rf_macro_score = f1_score(Y_testset, rf_prediction, average="macro") 

   print(f"F1 score estimate {rf_macro_score: .4f}")
   print(f"Classification report...")
   print(classification_report(Y_testset, rf_prediction))

   # SVM classifier, using a linear classifier
   svm_model = SVC(kernel='linear', random_state=random_seed)
   svm_model.fit(X_trainset, Y_trainset)
   # Linear performance
   svm_prediction = svm_model.predict(X_testset)
   svm_macro_score = f1_score(Y_testset, svm_prediction, average="macro") 

   print(f"F1 score estimate for SVM {svm_macro_score: .4f}")
   print(f"Classification report for SVM...")
   print(classification_report(Y_testset, svm_prediction))

   svm_all_predictions = svm_model.predict(X_testset)
   rf_all_predictions = rf_model.predict(X_testset)

   # 2. Build a clean comparison table for all speeches
   predictions_df = pd.DataFrame({
        "Row_Index": range(len(Y_testset)),
        "Actual_Party": Y_testset,
        "SVM_Predicted": svm_all_predictions,
        "RF_Predicted": rf_all_predictions
    })

   # Return label for party - use pandas to allow full output in a table and reset after
   pd.set_option('display.max_rows', None)
   unique_labels = np.unique(Y_partylabels).tolist()

   print(f"All Party Speech - Predictions Below ")
   print(f"{predictions_df.to_string(index=False)}")

   pd.reset_option('display.max_rows')
   return unique_labels

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
   # vectorize the speeches
   vectorise_speeches(df)
except ValueError as e:
  print(f"Unable to process {csv_to_process} {e}")