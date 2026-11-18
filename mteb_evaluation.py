from dotenv import load_dotenv
from mteb.models.model_meta import ModelMeta
import mteb
import bz2
import os
import pickle
import logging

from embedder import ZeroParameterEmbedder


# Define folder to save benchmark results in
logging.basicConfig(level=logging.INFO)
custom_cache = mteb.cache.ResultCache(cache_path="mteb_results")


if __name__ == "__main__":
    # Load HF token from .env file
    load_dotenv()
    os.environ["HF_TOKEN"] = os.getenv("HF-KEY")

    # Pick Benchmark tasks
    benchmark = mteb.get_benchmark("MTEB(eng, v2)")
    # Filter to only retrieval tasks
    tasks = [
        task for task in benchmark.tasks if task.metadata.type in [
            "Classification",
            "Clustering",
            "Reranking",
            "Summarization",
        ]]
    print(f"Found {len(tasks)} tasks")


    # Load model parts
    with bz2.open("tokenizer.pkl.bz2", "rb") as f:
        tokenizer = pickle.load(f)

    with bz2.open("token_dictionary.pkl.bz2", "rb") as f:
        token_dictionary = pickle.load(f)

    with bz2.open("token_vectors_skipgram_raw.pkl.bz2", "rb") as f:
        token_vectors_skipgram_raw = pickle.load(f)

    with bz2.open("token_vectors_skipgram_iwt.pkl.bz2", "rb") as f:
        token_vectors_skipgram_iwt = pickle.load(f)

    with bz2.open("token_vectors_skipgram.pkl.bz2", "rb") as f:
        token_vectors_skipgram = pickle.load(f)

    with bz2.open("infoweight_transformer.pkl.bz2", "rb") as f:
        infoweight_transformer = pickle.load(f)

    with bz2.open("wasserstein_raw_raw.pkl.bz2", "rb") as f:
        wasserstein_raw_raw = pickle.load(f)

    with bz2.open("wasserstein_iwt_raw.pkl.bz2", "rb") as f:
        wasserstein_iwt_raw = pickle.load(f)

    with bz2.open("wasserstein_raw_iwt.pkl.bz2", "rb") as f:
        wasserstein_raw_iwt = pickle.load(f)

    with bz2.open("wasserstein_iwt_iwt.pkl.bz2", "rb") as f:
        wasserstein_iwt_iwt = pickle.load(f)

    with bz2.open("wasserstein_w2v_raw.pkl.bz2", "rb") as f:
        wasserstein_w2v_raw = pickle.load(f)

    with bz2.open("wasserstein_w2v_iwt.pkl.bz2", "rb") as f:
        wasserstein_w2v_iwt = pickle.load(f)

    model_details = [
        ("uw/zpe_raw_raw", tokenizer, token_dictionary, token_vectors_skipgram_raw, None, wasserstein_raw_raw),
        ("uw/zpe_iwt_raw", tokenizer, token_dictionary, token_vectors_skipgram_iwt, None, wasserstein_iwt_raw),
        ("uw/zpe_w2v_raw", tokenizer, token_dictionary, token_vectors_skipgram, None, wasserstein_w2v_raw),
        ("uw/zpe_raw_iwt", tokenizer, token_dictionary, token_vectors_skipgram_raw, infoweight_transformer, wasserstein_raw_iwt),
        ("uw/zpe_iwt_iwt", tokenizer, token_dictionary, token_vectors_skipgram_iwt, infoweight_transformer, wasserstein_iwt_iwt),
        ("uw/zpe_w2v_iwt", tokenizer, token_dictionary, token_vectors_skipgram, infoweight_transformer, wasserstein_w2v_iwt),
    ]

    for details in model_details:
        # Create model
        model = ZeroParameterEmbedder("ZPE", "0.0.1", details[1], details[2], details[3], details[4], details[5])
        model.mteb_model_meta = ModelMeta(
            name=details[0],
            revision="0.0.1",
            languages=["eng-Latn"],
            n_parameters=None,
            n_embedding_parameters=3_000_000,
            loader= lambda **kwargs : None,
            max_tokens=None,
            embed_dim=128,
            release_date="2026-11-18",
            memory_usage_mb=None,
            open_weights=True,
            public_training_code="True",
            public_training_data="True",
            framework=[],
            similarity_fn_name="cosine",
            use_instructions=None,
            training_datasets=["wikitext"],
            license=None,
        )

        results = mteb.evaluate(
            model, 
            tasks=tasks,
            cache=custom_cache,
        )