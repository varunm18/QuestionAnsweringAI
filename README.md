# QuestionAnsweringAI

Adapted from my own implementation of Harvard's CS50AI's Week 6 Questions Problem Set

Produces an answer from the web given a question

### Process
1. Entity Linking and Word Sense Disambiguation AI is implemented with [Babely API](http://babelfy.org)
2. Keywords from the API response are used to load corresponding Wikipedia pages into a corpus
3. Question is then tokenized into words after filtering out stopwords and punctuation
4. Inverse Document Frequency(IDF) is then calculated for each word that shows up in the documents
5. The top files(number specified by user) is found given a words Term Frequency(TF) and IDF values (TF-IDF in short)
6. IDF is recalculated for sentences in the top documents
7. Sentences are ranked using their IDF to calculate Matching Word Measure, and, if there are ties, Query Term Density metric is used to return the top sentences(number specified by user)
