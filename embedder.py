import mteb
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.types import PromptType, Array, BatchedInput
from torch.utils.data import DataLoader
from vectorizers import NgramVectorizer

from mteb.models.model_meta import ModelMeta
from mteb.models.abs_encoder import AbsEncoder

class ZeroParameterEmbedder(AbsEncoder):
    def __init__(
            self,
            model_name,
            revision,
            tokenizer,
            token_dictionary,
            token_vectors,
            infoweight_transformer,
            wasserstein_vectorizer,
            device=None,
            internal_models_file: str | None = None,
            **kwargs
    ):
        self.model_name = model_name
        self.revision = revision

        self.tokenizer = tokenizer
        self.token_dictionary = token_dictionary
        self.token_vectors = token_vectors
        self.ngram_vectorizer = NgramVectorizer(token_dictionary=self.token_dictionary)
        self.infoweight_transformer = infoweight_transformer
        self.wasserstein_vectorizer = wasserstein_vectorizer

        self.x = 1

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs,
    ) -> Array:
        """Encodes the given sentences using the encoder.

        Args:
            inputs: The inputs to encode.
            task_metadata: The name of the task.
            hf_subset: The subset of the dataset.
            hf_split: The split of the dataset.
            prompt_type: The prompt type to use.
            **kwargs: Additional arguments to pass to the encoder.

        Returns:
            The encoded sentences.
        """
        all_text = [text for batch in inputs for text in batch["text"]]

        token_sequences = self.tokenizer.encode(all_text, out_type=str)
        
        token_counts = self.ngram_vectorizer.fit_transform(token_sequences)
        if self.infoweight_transformer is not None:
            token_counts = self.infoweight_transformer.transform(token_counts)
            
        document_vectors = self.wasserstein_vectorizer.transform(token_counts, vectors=self.token_vectors)

        return document_vectors.astype("float32")
