# Overview
- Word2Vec is the fundamental method of the distributed representation that vectorizes the word representation in multi-dimensional space. Word2Vec proposed two different methods: Continuous Bag of Words (CBoW) and Skip-Gram. CBoW is a method of predicting the target word from words around the target word called surrounding words. This project aims to implement CBoW and Skip-Gram word embedding methods.

# Brief description
- text_processing.py
> Output format
> - output: Tokenized result of a given text. (list)
- my_skipgram.py; my_cbow.py
> Output format
> - output: List of tensor of input tokens.

# Prerequisites
- argparse
- stanza
- spacy
- nltk
- gensim
- torch

# Parameters
- nlp_pipeline(str, defaults to "spacy"): Tokenization method (spacy, stanza, nltk, gensim).
- encoding(str, defaults to "bert"): Encoding method (onehot, skipgram, cbow, elmo, bert).
- emb_dim(int, defaults to 10): The size of word embedding.
- hidden_dim(int, defaults to 128): The hidden size of skipgram and cbow.
- window(int, defaults to 2): The size of window of skipgram and cbow.

# References
- Stanza: Qi, P., Zhang, Y., Zhang, Y., Bolton, J., & Manning, C. D. (2020). Stanza: A Python natural language processing toolkit for many human languages. arXiv preprint arXiv:2003.07082.
- Spacy: Matthew Honnibal and Ines Montani. 2017. spaCy 2: Natural language understanding with Bloom embeddings, convolutional neural networks and incremental parsing. To appear (2017).
- NLTK: Bird, Steven, Edward Loper and Ewan Klein (2009). Natural Language Processing with Python.  O'Reilly Media Inc.
- Gensim: Rehurek, R., & Sojka, P. (2010). Software framework for topic modelling with large corpora. In In Proceedings of the LREC 2010 workshop on new challenges for NLP frameworks.
- Word2vec: Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient estimation of word representations in vector space. arXiv preprint arXiv:1301.3781.
