# Modern Bag-of-Word Embedders

This repository holds code for training and testing a modern bag-of-words
embedding model. The embedder uses a collection non-transformer methods,
mostly centered around bags of words, to embedd documents into a shared space.
The workflow is

1. Tokenize the input documents.
2. Learn token vectors using TokenCooccurenceVectorizer and InformationWeightTransformer.
3. Make document vectors using the WassersteinVectorizer (with sinkhorn approximation).

The model(s) are trained on wikitext103, and the full model has about 3 millions parameters, extremely
small by modern standards. These techniques could be particularly helpful for either non-sequential data
(i.e. hypergraphs) or for low-resource settings (since less parameters generally requires less training data).

Evaluation of the method will be based on the [MTEB English Embedding
Leaderboard.](https://leaderboard.mteb.org/benchmark/MTEB(eng%2C%20v2))


