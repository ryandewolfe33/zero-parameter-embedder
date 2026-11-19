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


exclude_tasks = ["MindSmallReranking"]

if __name__ == "__main__":
    # Load HF token from .env file
    load_dotenv()
    os.environ["HF_TOKEN"] = os.getenv("HF-KEY")

    # Pick Benchmark tasks
    benchmark = mteb.get_benchmark("MTEB(eng, v2)")
    tasks = [t for t in benchmark.tasks if t.metadata.name not in exclude_tasks]

    print(f"Found {len(tasks)} tasks")
    for t in tasks:
        print(t.metadata.name, t.metadata.descriptive_stats["test"]["num_samples"])
    print()

    tokenizer_path = "tokenizer.pkl.bz2"
    token_dictionary_path = "token_dictionary.pkl.bz2"
    token_vectors_skipgram_raw_path = "token_vectors_skipgram_raw.pkl.bz2"
    token_vectors_skipgram_iwt_path = "token_vectors_skipgram_iwt.pkl.bz2"
    token_vectors_skipgram_path = "token_vectors_skipgram.pkl.bz2"
    infoweight_transformer_path = "infoweight_transformer.pkl.bz2"
    wasserstein_raw_raw_path = "wasserstein_raw_raw.pkl.bz2"
    wasserstein_raw_iwt_path = "wasserstein_iwt_raw.pkl.bz2"
    wasserstein_iwt_raw_path = "wasserstein_raw_iwt.pkl.bz2"
    wasserstein_iwt_iwt_path = "wasserstein_iwt_iwt.pkl.bz2"
    wasserstein_w2v_raw_path = "wasserstein_w2v_raw.pkl.bz2"
    wasserstein_w2v_iwt_path = "wasserstein_w2v_iwt.pkl.bz2"

    model_details = [
        ("uw/zpe_raw_raw", tokenizer_path, token_dictionary_path, token_vectors_skipgram_raw_path, None, wasserstein_raw_raw_path),
        ("uw/zpe_iwt_raw", tokenizer_path, token_dictionary_path, token_vectors_skipgram_iwt_path, None, wasserstein_iwt_raw_path),
        ("uw/zpe_w2v_raw", tokenizer_path, token_dictionary_path, token_vectors_skipgram_path, None, wasserstein_w2v_raw_path),
        ("uw/zpe_raw_iwt", tokenizer_path, token_dictionary_path, token_vectors_skipgram_raw_path, infoweight_transformer_path, wasserstein_raw_iwt_path),
        ("uw/zpe_iwt_iwt", tokenizer_path, token_dictionary_path, token_vectors_skipgram_iwt_path, infoweight_transformer_path, wasserstein_iwt_iwt_path),
        ("uw/zpe_w2v_iwt", tokenizer_path, token_dictionary_path, token_vectors_skipgram_path, infoweight_transformer_path, wasserstein_w2v_iwt_path),
    ]

    for details in model_details:
        # Check existing results in the local cache
        cached_results = custom_cache.load_results(models=[details[0]], tasks=tasks)
        cached_task_names = {
            task_res.task_name
            for model_res in cached_results.model_results
            for task_res in model_res.task_results
        }
        missing_tasks = [t for t in tasks if t.metadata.name not in cached_task_names]
        if len(missing_tasks) == 0:
            print("All tasks cached for", detials[0])
            continue

        print(len(missing_tasks), "tasks left to run for", details[0])

        with bz2.open(details[1], "rb") as f:
            tokenizer = pickle.load(f)
        with bz2.open(details[2], "rb") as f:
            token_dictionary = pickle.load(f)
        with bz2.open(details[3], "rb") as f:
            token_vectors = pickle.load(f)
        if details[4] is not None:
            with bz2.open(details[4], "rb") as f:
                infoweight = pickle.load(f)
        else:
            infoweight = None
        with bz2.open(details[5], "rb") as f:
            wasserstein = pickle.load(f)

        model = ZeroParameterEmbedder("ZPE", "0.0.1", tokenizer, token_dictionary, token_vectors, infoweight, wasserstein),
        # Create model
        model.metadata = ModelMeta(
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
            encode_kwargs = {'batch_size': 8192},
	)
