#NLP assessment template 2026

# Note: The template functions here and the dataframe format for structuring your solution is a suggested but not mandatory approach. You can use a different approach if you like, as long as you clearly answer the questions and communicate your answers clearly.

import string
import nltk
import spacy
from pathlib import Path
import pandas as pd

import os

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000

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
    author, and year"""
    novels = os.listdir(f"{Path.cwd()}/texts/novels")
    # vectorise list of novels
    novels = [n for n in novels]

    for val in novels:
        print(f"Novel title {val}")
        print(f"Title of novel is ", val)
        columns = ["text", "title", "author", "year"]
        df = pd.DataFrame(columns=columns)
        df.sort_values(by="year", inplace=True)
        print(df)
        pass
    return

def parse(df, store_path=Path.cwd() / "pickles", out_name="parsed.pickle"):
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting  DataFrame to a pickle file"""
    pass


def nltk_ttr(text):
    """Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize. Referred to: https://www.kaggle.com/code/kelixirr/tokenization-text-processing-in-nlp"""
    
    # Split all punctuation in text into its own tokens
    word_punctuation_tokenizer = nltk.WordPunctTokenizer()
    word_punctuation_tokens = word_punctuation_tokenizer.tokenize(text)
    # Clean punctuation - referred to: https://www.geeksforgeeks.org/python/string-punctuation-in-python/
    cleantxt = [t for t in word_punctuation_tokens if t not in string.punctuation]

    # Tokenise cleaned punctuation from txt
    tokenised_documents = nltk.word_tokenize(cleantxt)

    # Set types from tokenised documents
    types = set(tokenised_documents)
    #Use ttr to divide the number of words by the number of unique words. If there are no tokens, return 0.
    ttr = len(types).lower() / len(tokenised_documents) if tokenised_documents else 0
    #Define dictionary and now map values to the ttr values
    dictionary = text.dictionary(tokenised_documents)

    dict.mapping = {
        "title": ttr
    }

    return {
        "tokens": len(tokens),
        "types": len(types),
        "ttr": ttr
    }
    pass


def get_ttrs(df):
    """helper function to add ttr to a dataframe"""
    results = {}
    for i, row in df.iterrows():
        results[row["title"]] = nltk_ttr(row["text"])
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
    # print(df.head())
    # nltk.download("cmudict")
    # parse(df)
    # print(df.head())
    # print(get_ttrs(df))
    # print(get_fks(df))
    # df = pd.read_pickle(Path.cwd() / "pickles" /"name.pickle")
    # call functions for part (e) here.
