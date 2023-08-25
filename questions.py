import nltk
import sys
import os
import string
import math
import requests
import re

FILE_MATCHES = 1
SENTENCE_MATCHES = 1
WIKI_PAGES = 5
API_KEY = "152866d0-b5ab-459d-95b8-aa9da5228cdd"

def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")
    
    # Delete any existing file in corpus
    for file in os.listdir(os.path.join(os.getcwd(), 'corpus')):
        os.remove(os.path.join(os.getcwd(), 'corpus', file))
    
    # Prompt user for query
    text = input("Query: ")

    # Get API response of key words to query
    response = requests.get(f"https://babelfy.io/v1/disambiguate?text={text}&lang=EN&key={API_KEY}").json()
    
    page_count = 0
    # Parse API response to retrieve and store their wikipedia pages in corpus
    for item in response[:WIKI_PAGES+1]:
        if grp := re.search(r'^http://dbpedia.org/resource/(.+)$', item['DBpediaURL']):
            key_word = text[item['charFragment']['start']:item['charFragment']['end']+1]
            # print(key_word)
            # print(item['DBpediaURL'])
            # print(grp.group(1))
            # print()
            site = requests.get(
                'https://en.wikipedia.org/w/api.php',
                params={
                    'action': 'query',
                    'format': 'json',
                    'titles': grp.group(1),
                    'prop': 'extracts',
                    'explaintext': True,
                }).json()
            page = next(iter(site['query']['pages'].values()))
            with open(f'{os.path.join(sys.argv[1], key_word)}.txt', 'w') as file:
                file.write(page['extract'])
                page_count+=1

    # Indicate Wikipedia pages found
    if page_count==0:
        print("No Wikipedia pages found, please rephrase question to make it more specific")
    else:
        print(f'{page_count} Wikipedia pages found and parsed')

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Tokenize the input text
    query = set(tokenize(text))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(f'Answer: {match}')

def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    contents = dict()
    path = os.path.join(os.getcwd(), directory)
    for file in os.listdir(path):
        with open(os.path.join(directory, file), "r") as f:
            contents[file] = f.read()
    return contents

def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    tokenized = nltk.word_tokenize(document.lower())
    words = []
    for word in tokenized:
        if word not in list(string.punctuation)+nltk.corpus.stopwords.words("english"):
            words.append(word)
    return words

def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    idfs = dict()
    words = []

    for file in documents:
        for word in documents[file]:
            words.append(word)
    words_set = set(words)

    for word in words_set:
        count = 0
        for document in documents:
            if word in documents[document]:
                count+=1
        idfs[word] = math.log(float(len(documents))/float(count))
    return idfs

def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tdidfs = dict()

    for file in files:
        sum = 0
        for word in query:
            if word in idfs:
                sum += files[file].count(word)*idfs[word]
        tdidfs[file] = sum

    return_list = sorted(tdidfs, key=tdidfs.get)
    return_list.reverse()
    
    return return_list[:n]

def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    preference = dict()

    for sentence in sentences:
        sum = 0
        count = 0
        for word in query:
            if word in sentences[sentence]:
                sum += idfs[word]
                count+=1
        preference[sentence] = sum, count/len(sentence)
    
    return_list = sorted(preference, key=preference.get)
    return_list.reverse()

    return return_list[:n]

if __name__ == "__main__":
    main()
