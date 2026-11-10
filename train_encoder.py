from datasets import load_from_disk, load_dataset
from dotenv import load_dotenv
from infoweight import InformationWeightTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from vectorizers import TokenCooccurrenceVectorizer, NgramVectorizer, WassersteinVectorizer
import sentencepiece


from pathlib import Path
import gc
import io
import os
import pickle
import time


def format_timing(elapsed):
    if elapsed < 1e-3:
        # Microseconds
        time_str = f"{elapsed * 1e6:.2f} μs"
    elif elapsed < 1:
        # Milliseconds
        time_str = f"{elapsed * 1e3:.2f} ms"
    elif elapsed < 60:
        # Seconds
        time_str = f"{elapsed:.2f} s"
    else:
        # Minutes and Seconds
        minutes, seconds = divmod(elapsed, 60)
        time_str = f"{int(minutes)}m {seconds:.2f}s"
    return time_str


def dist_dr(skipgrams):
    # Row normalize token vectors into distributions and decrease precision
    skipgram_distributions = normalize(skipgrams, norm="l1", axis=1).astype("float32")
    svd = TruncatedSVD(n_components=100, random_state=42)
    token_vectors_skipgram = svd.fit_transform(skipgram_distributions.sqrt())
    # SQRT makes euclidean equal to Hellinger
    return token_vectors_skipgram


def fit_wasserstein(token_counts, token_vectors, path):
    start = time.perf_counter()
    wasserstein = WassersteinVectorizer(method="LOT_sinkhorn",
                                        memory_size="1G",
                                        reference_size=25,
                                        random_state=42)
    wasserstein.fit(token_counts, vectors=token_vectors)
    end = time.perf_counter()
    print(f"Finished in {format_timing(end - start)}\n")
    with open(path, "wb") as f:
        pickle.dump(wasserstein, f)


print("Tokenizer Part")
data_path = "wikitext-103-v1"
tokenizer_path = Path("tokenizer.pkl")
token_sequences_path = Path("token_sequences.pkl")
token_dict_path = Path("token_dictionary.pkl")
token_counts_path = Path("token_counts.pkl")
infoweight_transformer_path = Path("infoweight_transformer.pkl")

if (
    Path(data_path).is_dir()
    and tokenizer_path.is_file()
    and token_sequences_path.is_file()
    and token_dict_path.is_file()
    and token_counts_path.is_file()
    and infoweight_transformer_path.is_file()
):
    print("All tokenizations step already cached.")
else:
    def tokenizer_steps(data_path, tokenizer_path, token_sequences_path, token_dict_path, token_counts_path, infoweight_transformer_path):
        # Put as a function to make sure memory does not leak
        if Path(data_path).is_dir():
            print("Loading dataset")
            data = load_from_disk("wikitext-103-v1")
            print("Dataset loaded\n")
        else:
            print("Downloading dataset")
            load_dotenv()
            api_key = os.getenv("HF-KEY")
            data = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="train", token=api_key)
            data.save_to_disk("wikitext-103-v1")

        print("Pruning dataset")
        start = time.perf_counter()
        # Get a list of training tests
        text_list = list(data["text"])
        # Prune almost empty documents (<30 characters, arbitray threshold)
        pruned_text_list = [t for t in text_list if len(t) > 100]
        end = time.perf_counter()
        print(f"{len(pruned_text_list)} documents remain. {format_timing(end - start)}\n")

        print("Train Tokenizer")
        start = time.perf_counter()
        model_buffer = io.BytesIO()
        sentencepiece.SentencePieceTrainer.Train(
            sentence_iterator=(x for x in pruned_text_list), model_writer=model_buffer, vocab_size=10000
        )
        tokenizer = sentencepiece.SentencePieceProcessor(model_proto=model_buffer.getvalue())
        end = time.perf_counter()
        print(f"Finished in {format_timing(end - start)}\n")
        with open(tokenizer_path, "wb") as f:
            pickle.dump(tokenizer, f)

        token_sequences = tokenizer.encode(pruned_text_list, out_type=str)
        with open(token_sequences_path, "wb") as f:
            pickle.dump(token_sequences, f)

        unique_tokens = list(set.union(*[set(s) for s in token_sequences]))
        token_dictionary = {t:i for i,t in enumerate(unique_tokens)}
        with open(token_dict_path, "wb") as f:
            pickle.dump(token_dictionary, f)
        print(f"{len(token_dictionary)} unique tokens in training corpus\n")

        print("Train Ngram Vectorizer")
        start = time.perf_counter()
        counter = NgramVectorizer(token_dictionary=token_dictionary)
        token_counts = counter.fit_transform(token_sequences)
        end = time.perf_counter()
        print(f"Finished in {format_timing(end - start)}\n")
        with open(token_counts_path, "wb") as f:
            pickle.dump(token_counts, f)
        print(f"Token counts: {token_counts.shape} with {token_counts.nnz} entries")

        print("Fit IWT to token counts")
        start = time.perf_counter()
        infoweight_transformer = InformationWeightTransformer()
        infoweight_transformer.fit(token_counts)
        end = time.perf_counter()
        print(f"Finished in {format_timing(end - start)}\n")
        with open(infoweight_transformer_path, "wb") as f:
            pickle.dump(infoweight_transformer , f)

    # Saves files but does not return anything
    tokenizer_steps(data_path, tokenizer_path, token_sequences_path, token_dict_path, token_counts_path, infoweight_transformer_path)



gc.collect()
# End of tokenization part.


# Token vectorization
print("Token Vectorization")
token_vectors_skipgram_raw_path = Path("token_vectors_skipgram_raw.pkl")
token_vectors_skipgram_iwt_path = Path("token_vectors_skipgram_iwt.pkl")

if token_vectors_skipgram_raw_path.is_file() and token_vectors_skipgram_iwt_path.is_file():
    print("Token vectorization already cached.")
else:
    with open(token_sequences_path, "rb") as f:
        token_sequences = pickle.load(f)
    with open(token_dict_path, "rb") as f:
        token_dictionary = pickle.load(f)

    def token_vectorization(token_sequences, token_dictionary, token_vectors_skipgram_raw_path, token_vectors_skipgram_iwt_path):
        print("Training Skipgrams")
        start = time.perf_counter()
        skipgram_vectorizer = TokenCooccurrenceVectorizer(token_dictionary=token_dictionary)
        skipgrams = skipgram_vectorizer.fit_transform(token_sequences)
        end = time.perf_counter()
        print(f"Finished in {format_timing(end - start)}\n")

        print("Dimension reduce skipgram distributions")
        start = time.perf_counter()
        token_vectors_skipgram_raw = dist_dr(skipgrams)
        end = time.perf_counter()
        print(f"Finished in {format_timing(end - start)}\n")
        with open(token_vectors_skipgram_raw_path, "wb") as f:
            pickle.dump(token_vectors_skipgram_raw, f)

        print("IWT Skipgrams")
        start = time.perf_counter()
        iwt = InformationWeightTransformer()
        iwt_skipgrams = iwt.fit_transform(skipgrams).tocsr()
        token_vectors_skipgram_iwt = dist_dr(iwt_skipgrams)
        end = time.perf_counter()
        print(f"Finished in {format_timing(end - start)}\n")
        with open(token_vectors_skipgram_iwt_path, "wb") as f:
            pickle.dump(token_vectors_skipgram_iwt, f)

    token_vectorization(token_sequences, token_dictionary, token_vectors_skipgram_raw_path, token_vectors_skipgram_iwt_path)

gc.collect()
# End token vectorization


# Start document vectorization
# Do each vectorizer one at a time and gc after each.
wasserstein_raw_raw_path = Path("wasserstein_raw_raw.pkl")
if wasserstein_raw_raw_path.is_file():
    print("Wasserstein Raw Raw already exists, skip.")
else:
    with open(token_counts_path, "rb") as f:
        token_counts = pickle.load(f)
    with open(token_vectors_skipgram_raw_path, "rb") as f:
        token_vectors_skipgram_raw = pickle.load(f)
    print("Fit Wasserstein Raw Raw")
    fit_wasserstein(token_counts, token_vectors_skipgram_raw, wasserstein_raw_raw_path)
    del token_counts
    del token_vectors_skipgram_raw
gc.collect()


# wasserstein_iwt_raw_path = Path("wasserstein_iwt_raw.pkl")
# if wasserstein_iwt_raw_path.is_file():
#     print("Wasserstein Raw Raw already exists, skip.")
# else:
#     with open(token_counts_path, "rb") as f:
#         token_counts = pickle.load(f)
#     with open(token_vectors_skipgram_iwt_path, "rb") as f:
#         token_vectors_skipgram_iwt = pickle.load(f)
#     print("Fit Wasserstein IWT Raw")
#     fit_wasserstein(token_counts, token_vectors_skipgram_iwt, wasserstein_iwt_raw_path)
#     del token_counts
#     del token_vectors_skipgram_iwt
# gc.collect()


# wasserstein_raw_iwt_path = Path("wasserstein_raw_iwt.pkl")
# if wasserstein_raw_iwt_path.is_file():
#     print("Wasserstein Raw IWT already exists, skip.")
# else:
#     with open(token_counts_path, "rb") as f:
#         token_counts = pickle.load(f)
#     with open(infoweight_transformer_path, "rb") as f:
#         iwt = pickle.load(f)
#     iwt_token_counts = iwt.transform(token_counts)
#     del iwt
#     with open(token_vectors_skipgram_raw_path, "rb") as f:
#         token_vectors_skipgram_raw = pickle.load(f)
#     print("Fit Wasserstein Raw IWT")
#     fit_wasserstein(iwt_token_counts, token_vectors_skipgram_raw, wasserstein_raw_iwt_path)
#     del iwt_token_counts
#     del token_vectors_skipgram_raw
# gc.collect()


# wasserstein_iwt_iwt_path = Path("wasserstein_iwt_iwt.pkl")
# if wasserstein_iwt_iwt_path.is_file():
#     print("Wasserstein IWT IWT already exists, skip.")
# else:
#     with open(token_counts_path, "rb") as f:
#         token_counts = pickle.load(f)
#     with open(infoweight_transformer_path, "rb") as f:
#         iwt = pickle.load(f)
#     iwt_token_counts = iwt.transform(token_counts)
#     del iwt
#     with open(token_vectors_skipgram_iwt_path, "rb") as f:
#         token_vectors_skipgram_iwt = pickle.load(f)
#     print("Fit WassersteinIWT IWT")
#     fit_wasserstein(iwt_token_counts, token_vectors_skipgram_iwt, wasserstein_iwt_iwt_path)
#     del iwt_token_counts
#     del token_vectors_skipgram_iwt
# gc.collect()
