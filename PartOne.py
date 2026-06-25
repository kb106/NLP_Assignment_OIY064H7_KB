#NLP assessment template 2026

# Note: The template functions here and the dataframe format for structuring your solution is a suggested but not mandatory approach. You can use a different approach if you like, as long as you clearly answer the questions and communicate your answers clearly.
from collections import defaultdict
import pickle
import re

import nltk
import spacy
from spacy.tokens import DocBin
from pathlib import Path
import pandas as pd

import os
from curses.ascii import isdigit

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

    Implements Flesch Formula:
        Reading Ease score = 206.835 - (1.015 × ASL) - (84.6 × ASW)

        ASL = Avg Sentence Length
        ASW = Avg Word Length
        
    Args:
        text (str): The text to analyze.
        d (dict): A dictionary of syllables per word.

    Returns:
        float: The Flesch-Kincaid Grade Level of the text. (higher grade is more difficult)
    """
    #  Referred to WK4 - Sentences and https://datawarrior.wordpress.com/2016/03/29/flesch-kincaid-readability-measure/
    # Set fkresults as the dictionary to add to
    
    fkresults = {}
    num_syllables = 0
    total_syllables = 0

    # find the length of words via the tokenized text and filter out punctuation
    tokens = [token.lower() for token in nltk.word_tokenize(text) if token.isalpha()]
    num_words = len(tokens)
    # find the the length of sentences based on the number of sentences
    num_sentences = len(nltk.sent_tokenize(text))
    # initialise and get the syllables for each word via Carnegie-Mellon University (CMU) dictionary
    sylldict = nltk.corpus.cmudict.dict()

    for word in tokens:
        # Handle cases where the number of words or sentences returns NULL
        if num_words == 0 or num_sentences == 0:
            return 0.0
    
        else:
            # count number of syllables in a word by calling the function below
            total_syllables += count_syl(word, sylldict)

    # Formula for Averages and FK
    avg_words_sentence = num_words / num_sentences if num_sentences > 0 else 0 # avergae out the number of words in each sentence
    avg_syll_word = total_syllables / num_words if num_words > 0 else 0 # average out the number of syllables per word
 
    # Formula for reading Ease score =  (0.39 × ASL) + (11.8 × ASW) - 15.59  
    # reference: https://readabilityformulas.com/learn-how-to-use-the-flesch-kincaid-grade-level/
    fkresults =  (0.39 * avg_words_sentence) + (11.8 * avg_syll_word) - 15.59
  
    # Some novel text return lower complexity scores for readability when in fact the prose is dense, and the presence of single syllables per word does not always accurately measure readability 
    return fkresults 
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
    num_syllables = 0
    getword = word.lower().strip()

    try:
        if not getword or not getword.isalpha():
            return 0

        if getword in d:
            # Referenced syllable count pattern from: https://gist.github.com/drinks/2483508
            phonemes =  d[getword][0]
            num_syllables += len([i for i in phonemes if i[-1].isdigit()])

        else:
            # Handle words missing from dictionary by counting the instances of vowels in a word (fallback ONLY)...    
            vowelcount = re.findall(f'r[aeiou]+', getword)
            vowel_cluster = len(vowelcount)
            num_syllables =  vowel_cluster

    except KeyError:
           print(f"Unable to process count of syllables in word")
        
    return num_syllables
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
    # use dataframe with spacy nlp to retrieve text field contents in a list, also set the nlp parsed output to a max length - referred to: https://spacy.io/usage/saving-loading
    # max_length for nlp referred to: https://spacy.io/api/language
    nlp.max_length = 3000000
    parsed_docs = list(nlp.pipe(df["text"].astype(str), disable=["lemmatizer", "ner"]))
    doc_bin = DocBin(attrs=["ORTH", "IS_ALPHA" ,"LEMMA"])
    # Define the path to output too
    store_path.mkdir(parents=True, exist_ok=True)
    doc_text_path = store_path / out_name 

    # Loop to get texts into objects
    for doc in parsed_docs:
        # create document object and convert to bytes to serialise
        doc_bin.add(doc)
        
    # define column to add text object too and use DocBin to setup the doc object we need
    df['parsed_docs'] = parsed_docs

    #Section for saving and reading each novel text to/from pickle file# 
    # Assign doc object to pickle file - referred to: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_pickle.html
    df.to_pickle(str(doc_text_path))

    print(f"{doc_text_path} file successfully saved")

    for i, row in df.iterrows():
        title = row["title"]
        noveltext = row["parsed_docs"][:nlp.max_length]
        print(
            f"\n-----Novel Title  {title}"
            f"\n {noveltext}... "
        ) # Prints lemmas of the first 500 words

    return df
    pass

def nltk_ttr(text):
    """Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize. Referred to: https://www.kaggle.com/code/kelixirr/tokenization-text-processing-in-nlp"""
    # Use spacy, reference: https://spacy.io/models/en and Lab3 sentences from NLP course 
    # Reference global variable for spacy to extract and tokenise text
    output_text = nltk.word_tokenize(text)
    # For TTR ratio we want to measure the value returned between 1 (highly diverse and unique set of words) and 0 or higher repetition
    # Vectorize token call, ensure text is not case-sensitive and use is_alpha to filter out punctuation before parsing texts to get TTR
    tokens = [token.lower() for token in output_text if token.isalpha]

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
    print(f"Novel title and FKS score: {results}")
    return results

#.. add functions for part (e) here
def get_subjects(df):
 '''List of top 10 most common subject areas in text within novels'''
 all_text = parse(df)

# iterate over txt rows
 for i, row in all_text.iterrows():
     novel_doc = row["parsed_docs"]
     novel_parsed = nlp(novel_doc)

      #for tok in novel_parsed:
        #Get top 10 subjects by summarising text themes by using hugging face

def get_pmi_subject(parsed_df):
 '''PMI list of verbs most likely to occur where the subject 'he' or 'she' is used in text within novels'''
 # Referred to: NLP Lab3 ipynb file section 5 on PMI
 corpus_size = len(parsed_df)

 # Most common VERBS where the PRONOUN he is present
 # Set array to hold the resulting pair fo verbs to pronoun he 
 verb_counts = defaultdict(int)
 pron_counts = defaultdict(int)
 pair_counts = defaultdict(int)

 # iterate over txt batches - using call to column directly to return the text via the dataFrame
 for doc in parsed_df["parsed_docs"]:
     for tok in doc:
        # Ensuring that correct pronoun can be found following a verb by using token - child (Rule: verb always precedes pronoun, which is the child)
        if tok.pos_ in ["PRON"] and tok.text.lower() in ["he" or "she"]:
            pronoun = tok.text.lower()
            pron_counts[pronoun] += 1
            # Find VERBS and match pronouns - 'he'
            potential_verb = tok.head
            if potential_verb.pos_ == "AUX":
                potential_verb = potential_verb.head

            if potential_verb.pos_ == "VERB":
                # set verbs and pronouns found to lower case to capture all possible matches
                verbword = potential_verb.lemma_.lower()
                #pronounword = potential_verb.text.lower()
                verb_counts[verbword] += 1
                # Append pairs of verbs to pronoun
                pair_counts[(verbword, pronoun)] += 1
                # Store all resulting verbs from pronoun 'he'
 if not pair_counts:
    print(f"No pairs found")
    return [], []

        # TO DO  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
        # Now complete PMI calculations - referred to Lab3 example and https://medium.com/@fr4nk/understanding-pointwise-mutual-information-a-beginners-guide-dcfed0f83ff2
        # Formula: PMI(x, y) = log2(P(x,y) / (P(x) * P(y)))
        # where:
        # x and y   are the two words being analyzed (common verb, or x, and he or she, or y)
        # P(x,y)    is the probability of both x and y occurring together in a text corpus
        # P(x)      is the probability of x occurring in the corpus
        # P(y)      is the probability of y occurring in the corpus
        # verb_word_he_total = "Verb_word_he" / corpus_size
        # verb_word_she_total = "Verb word she" / corpus_size
        # word_pair_total = word_pair_counts_he / corpus_size
        # PMI = log2(word_pair_total) / ("Verb_word_she" )
 # print verb - pron pairs
 return pair_counts, verb_counts, pron_counts

if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    path = Path.cwd() / "texts" / "novels"
    print(path)
    df = read_novels(path) # this line will fail until you have completed the read_novels function above.
    get_ttrs(df)
    nltk.download("cmudict")
    get_fks(df)
    # print(df.head())
    parsed_df = parse(df)
    get_pmi_subject(df)
    # print(df.head())
    # print(get_ttrs(df))
    # print(get_fks(df))
    df = pd.read_pickle(Path.cwd() / "pickles" /"parsed.pickle")
    # call functions for part (e) here.
