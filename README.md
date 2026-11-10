# Zero Parameter Embedder

This repository holds code for defining and testing a zero parameter
embedding model on mteb semantic similarity tasks. The embedder uses
a collection non-transformer methods, mostly centered around bags
of words, to embedd documents into a shared space. The workflow is

1. Tokenize the input documents using BytePairEncoding.
2. Produce token vectors using SkipgramVectorizer and InformationWeightTransformer.
3. Make document vectors using the WassersteinVectorizer (or a faster approximation).
4. Reduce dimension using SVD/MDS/UMAP to something reasonable (but possibly still
in the 100s).

There are no neural-networks in any step, so this pipeline acts as a
zero paremeter embedding model (hence the repository name). However,
there are lots of hyperparameters that we will generally ignore, just
like the many neural-embedders. Having no parameters there is no
training, so we avoid (legal) questions and hassle of acquiring training
data, and avoid concerns of poisoning. It also means the embedder is
useful when there is minimal data available, for example in low resource
languages, or non-text data. The goal is not to produce a state-of-the-art
embedding method, but to see how existing and under-development mathematically
grounded methods compare to modern neural embedders.

Evaluation of the method will be based on the [MTEB English Embedding
Leaderboard.](https://leaderboard.mteb.org/benchmark/MTEB(eng%2C%20v2))


