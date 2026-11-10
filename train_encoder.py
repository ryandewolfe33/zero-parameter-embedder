from datasets import load_from_disk, load_dataset
from dotenv import load_dotenv
from infoweight import InformationWeightTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from vectorizers import TokenCooccurrenceVectorizer, NgramVectorizer, WassersteinVectorizer
import sentencepiece


from pathlib import Path
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


if Path("wikitext-103-v1").is_dir():
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

tokenizer_path = Path("tokenizer.pkl")
if tokenizer_path.is_file():
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    print("Loaded Tokenizer")
else:
    print("Train Tokenizer")
    start = time.perf_counter()
    model_buffer = io.BytesIO()
    sentencepiece.SentencePieceTrainer.Train(
        sentence_iterator=(x for x in pruned_text_list), model_writer=model_buffer, vocab_size=10000
    )
    tokenizer = sentencepiece.SentencePieceProcessor(model_proto=model_buffer.getvalue())
    token_sequences = tokenizer.encode(pruned_text_list, out_type=str)
    end = time.perf_counter()
    print(f"Finished in {format_timing(end - start)}\n")
    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)

token_dict_path = Path("token_dictionary.pkl")
if token_dict_path.is_file():
    with open(token_dict_path, "rb") as f:
        token_dictionary = pickle.load(f)
    print("Loaded token dictionary")
else:
    unique_tokens = list(set.union(*[set(s) for s in token_sequences]))
    token_dictionary = {t:i for i,t in enumerate(unique_tokens)}
    with open(token_dict_path, "wb") as f:
        pickle.dump(token_dictionary, f)
print(f"{len(token_dictionary)} unique tokens in training corpus\n")

token_vectors_skipgram_raw_path = Path("token_vectors_skipgram_raw.pkl")
token_vectors_skipram_iwt_path = Path("token_vectors_skipgram_iwt.pkl")
if token_vectors_skipgram_raw_path.is_file() and token_vectors_skipram_iwt_path.is_file():
    with open(token_vectors_skipgram_raw_path, "rb") as f:
        token_vectors_skipgram_raw = pickle.load(f)
    with open(token_vectors_skipram_iwt_path, "rb") as f:
        token_vectors_skipgram_iwt = pickle.load(f)
    print("Loaded skipgram token distribution vectors")
else:
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
    with open(token_vectors_skipram_iwt_path, "wb") as f:
        pickle.dump(token_vectors_skipgram_iwt, f)

token_counts_path = Path("token_counts.pkl")
if token_counts_path.is_file():
    with open(token_counts_path, "rb") as f:
        token_counts = pickle.load(f)
    print("Loaded token counts")
else:
    print("Train Ngram Vectorizer")
    start = time.perf_counter()
    counter = NgramVectorizer(token_dictionary=token_dictionary)
    token_counts = counter.fit_transform(token_sequences)
    print(f"Finished in {format_timing(end - start)}\n")
    with open(token_counts_path, "wb") as f:
        pickle.dump(token_counts, f)
print(f"Token counts: {token_counts.shape} with {token_counts.nnz} entries")

wasserstein_raw_raw_path = Path("wasserstein_raw_raw.pkl")
if wasserstein_raw_raw_path.is_file():
    print("Wasserstein Raw Raw already exists, skip.")
else:
    print("Fit Wasserstein Raw Raw")
    start = time.perf_counter()
    wasserstein_raw_raw = WassersteinVectorizer(method="LOT_sinkhorn", memory_size="4G", random_state=42)
    wasserstein_raw_raw.fit(token_counts, vectors=token_vectors_skipgram_raw)
    end = time.perf_counter()
    print(f"Finished in {format_timing(end - start)}\n")
    with open(wasserstein_raw_raw_path, "wb") as f:
        pickle.dump(wasserstein_raw_raw, f)

wasserstein_iwt_raw_path = Path("wasserstein_iwt_raw.pkl")
if wasserstein_iwt_raw_path.is_file():
    print("Wasserstein Raw Raw already exists, skip.")
else:
    print("Fit Wasserstein IWT Raw")
    start = time.perf_counter()
    wasserstein_iwt_raw = WassersteinVectorizer(method="LOT_sinkhorn", memory_size="4G", random_state=42)
    wasserstein_iwt_raw.fit(token_counts, vectors=token_vectors_skipgram_iwt)
    end = time.perf_counter()
    print(f"Finished in {format_timing(end - start)}\n")
    with open(wasserstein_iwt_raw_path, "wb") as f:
        pickle.dump(wasserstein_iwt_raw, f)

infoweight_transformer_path = Path("infoweight_transformer.pkl")
if infoweight_transformer_path.is_file():
    with open(infoweight_transformer_path, "rb") as f:
        infoweight_transformer = pickle.load(f)
    print("Loaded token counts IWT")
else:
    print("Fit IWT to token counts")
    start = time.perf_counter()
    infoweight_transformer = InformationWeightTransformer()
    infoweight_transformer.fit(token_counts)
    end = time.perf_counter()
    print(f"Finished in {format_timing(end - start)}\n")
    with open(infoweight_transformer_path, "wb") as f:
        pickle.dump(infoweight_transformer , f)

iwt_token_counts = infoweight_transformer.transform(token_counts).tocsr()
print("IWT'd token counts\n")

wasserstein_raw_iwt_path = Path("wasserstein_raw_iwt.pkl")
if wasserstein_raw_iwt_path.is_file():
    print("Wasserstein Raw IWT already exists, skip.")
else:
    print("Fit Wasserstein Raw IWT")
    start = time.perf_counter()
    wasserstein_raw_iwt = WassersteinVectorizer(method="LOT_sinkhorn", memory_size="4G", random_state=42)
    wasserstein_raw_iwt.fit(iwt_token_counts, vectors=token_vectors_skipgram_raw)
    end = time.perf_counter()
    print(f"Finished in {format_timing(end - start)}\n")
    with open(wasserstein_raw_iwt_path, "wb") as f:
        pickle.dump(wasserstein_raw_iwt, f)

wasserstein_iwt_iwt_path = Path("wasserstein_iwt_iwt.pkl")
if wasserstein_iwt_iwt_path.is_file():
    print("Wasserstein IWT IWT already exists, skip.")
else:
    print("Fit Wasserstein IWT IWT")
    start = time.perf_counter()
    wasserstein_iwt_iwt = WassersteinVectorizer(method="LOT_sinkhorn", memory_size="4G", random_state=42)
    wasserstein_iwt_iwt.fit(iwt_token_counts, vectors=token_vectors_skipgram_iwt)
    end = time.perf_counter()
    print(f"Finished in {format_timing(end - start)}\n")
    with open(wasserstein_iwt_iwt_path, "wb") as f:
        pickle.dump(wasserstein_iwt_iwt, f)
