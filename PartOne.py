#NLP assessment template 2026

# Note: The template functions here and the dataframe format for structuring your solution is a suggested but not mandatory approach. You can use a different approach if you like, as long as you clearly answer the questions and communicate your answers clearly.

import datetime
import string
import nltk
import spacy
from pathlib import Path
import pandas as pd
import os

try:
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 2000000
except OSError:
    raise(
            "Error when loading spacy model: 'en_core_web_sm' not found. Install it with:\n"
            "python -m spacy download en_core_web_sm - {ValueError}"
        )


def fk_level(text, d):
    """Returns the Flesch-Kincaid Grade Level of a text (higher grade is more difficult).
    Requires a dictionary of syllables per word.

    Args:
        text (str): The text to analyze.
        d (dict): A dictionary of syllables per word.

    Returns:
        float: The Flesch-Kincaid Grade Level of the text. (higher grade is more difficult)
    """
    pass

def count_syl(word, d):
    """Counts the number of syllables in a word given a dictionary of syllables per word.
    if the word is not in the dictionary, syllables are estimated by counting vowel clusters

    Args:
        word (str): The word to count syllables for.
        d (dict): A dictionary of syllables per word.

    Returns:
        int: The number of syllables in the word.
    """
    pass

def read_novels(path=Path.cwd() / "texts" / "novels"):
    """Reads texts from a directory of .txt files and returns a DataFrame with the text, title,
    author, and year, reference: https://builtin.com/data-science/python-list-files-in-directory, 
    as well as a local instance of Ollama API to clean up code"""
    # Initialise dataframe rows
    rows = []

    # handle condition where novel file is not a text file
    for file in os.listdir(path):
        if not file.endswith(".txt"):
            continue
        novels_path = path / file

        # Read text    
        txt = novels_path.read_text(encoding="utf-8")

        # Start with a split function to extract the columns from each novel
        split_filename = novels_path.stem
        filename_parts = split_filename.split('-')

        # Get all parts of the file according to column names, we need at least three parts
        if len(filename_parts) == 3:
            # assign columns to the parts
            title, author, year = filename_parts 
            # Use a try/catch to get the year for sorting the dataframe (df) and assign data type to col
            try:
                year = int(year)
            except ValueError:
                year = "None"
        else:
            title, author, year = filename_parts[0], filename_parts[1], filename_parts[2]

        # create dataframe using pandas (pd) - also clean special chars from the text and title
        rows.append({
            "text": txt.replace('\n',' '),
            "title": title.replace('_',' '),
            "author": author,
            "year": year
        }) 
    # create dataframe using pandas (pd)
    df = pd.DataFrame(rows, columns=["text", "title", "author", "year"])
    df_sorted = df.sort_values(by="year", ascending=True)
    print(df_sorted)

    pass
    return df

def parse(df, store_path=Path.cwd() / "pickles", out_name="parsed.pickle"):
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting  DataFrame to a pickle file"""
    pass

def nltk_ttr(text):
    """Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize. Referred to: https://www.kaggle.com/code/kelixirr/tokenization-text-processing-in-nlp"""
    # Use spacy, reference: https://spacy.io/models/en
    # Reference global variable for spacy to extract and tokenise text
    output_text = nlp(text)
    # For TTR ration we want to measure the value returned between 1 (highly diverse and unique set of words) and 0 or higher repetition
    # Vectorize token call, ensure text is not case-sensitive and use is_alpha to filter out punctuation before parsing texts to get TTR
    tokens = [token.text.lower() for token in output_text if token.is_alpha]

    # Handle cases where there are no tokens and return null 
    if not tokens:
        return 0.0
    else:
        # Set unique tokens (words) - or more diverse words
        unique_instances =  set(tokens)      
        #Use ttr to divide the number of words by the number of unique words. If there are no tokens, return 0.
        ttr = len(unique_instances) / len(tokens) if tokens else 0

    return ttr
    pass

def get_ttrs(df):
    """helper function to add ttr to a dataframe"""
    # Define dictionary and now map values to the ttr values
    # map title of novel to ttr using dictionary
    results = {}
    
    for i, row in df.iterrows():
        results[row["title"]]=nltk_ttr(row["text"])
    print(f"Novel title and TTR value: {results}") 
    return results

def get_fks(df):
    """helper function to add fk scores to a dataframe"""
    results = {}
    cmudict = nltk.corpus.cmudict.dict()
    for i, row in df.iterrows():
        results[row["title"]] = round(fk_level(row["text"], cmudict), 4)
    return results

#.. add functions for part (e) here

if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    path = Path.cwd() / "texts" / "novels"
    print(path)
    df = read_novels(path) # this line will fail until you have completed the read_novels function above.
    get_ttrs(df)
    # print(df.head())
    # nltk.download("cmudict")
    # parse(df)
    # print(df.head())
    # print(get_ttrs(df))
    # print(get_fks(df))
    # df = pd.read_pickle(Path.cwd() / "pickles" /"name.pickle")
    # call functions for part (e) here.
