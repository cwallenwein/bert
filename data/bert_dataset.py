from data.context_length_splitter import ContextLengthSplitter
from data.nsp_data_processor import NextSentencePredictionDataProcessor
from data.mlm_data_processor import MaskedLanguageModelingDataProcessor
from data.util import set_progress_bar
from data.save_and_load import get_dataset, get_tokenizer, save_dataset, dataset_exists


class BertDataset:
    # TODO: give option that both sentences for nsp are not actual sentences but just spans of contiguous text
    # TODO: compare performance

    @staticmethod
    def split_into_context_length(context_length: int, verbose: bool = True, load_if_exists: bool = True):
        # generate split_by_context_length dataset or load it
        with set_progress_bar(verbose):
            if dataset_exists(split_by_context_length=True, context_length=context_length) and load_if_exists:
                return get_dataset(split_by_context_length=True, context_length=context_length)
            else:
                dataset = get_dataset(raw=True)
                tokenizer = get_tokenizer()
                context_length_splitter = ContextLengthSplitter(tokenizer=tokenizer)
                dataset = context_length_splitter(
                    dataset=dataset,
                    context_length=context_length
                )
                save_dataset(dataset, split_by_context_length=True, context_length=context_length, overwrite=True)
                return dataset

    @staticmethod
    def apply_mlm_and_nsp(
        num_samples: int, preprocessed_name: str, context_length: int = None, verbose: bool = True, load_if_exists: bool = True
    ):
        BertDataset.split_into_context_length(context_length=context_length, verbose=verbose)
        with set_progress_bar(verbose):
            if dataset_exists(
                preprocessed=True, preprocessed_name=preprocessed_name, context_length=context_length
            ) and load_if_exists:
                return get_dataset(preprocessed=True, preprocessed_name=preprocessed_name, context_length=context_length)
            else:
                if dataset_exists(split_by_context_length=True, context_length=context_length):
                    dataset = get_dataset(split_by_context_length=True, context_length=context_length)
                    tokenizer = get_tokenizer()
                    nsp_data_collator = NextSentencePredictionDataProcessor(
                        tokenizer=tokenizer
                    )
                    dataset = nsp_data_collator(
                        dataset=dataset,
                        num_samples=num_samples,
                        context_length=context_length,
                        verbose=verbose,
                    )
                    assert len(dataset) == num_samples, f"{len(dataset)} != {num_samples}"

                    mlm_data_collator = MaskedLanguageModelingDataProcessor(
                        tokenizer=tokenizer,
                    )

                    dataset = mlm_data_collator(
                        dataset=dataset
                    )
                    assert len(dataset) == num_samples, f"{len(dataset)} != {num_samples}"
                    save_dataset(
                        dataset,
                        preprocessed=True,
                        preprocessed_name=preprocessed_name,
                        context_length=context_length,
                        overwrite=True
                    )
                    return dataset
                else:
                    raise Exception()

    @staticmethod
    def load(preprocessed_name: str, context_length: int = None, verbose: bool = True):
        # load mlm and nsp
        with set_progress_bar(verbose):
            if dataset_exists(preprocessed=True, preprocessed_name=preprocessed_name, context_length=context_length):
                return get_dataset(preprocessed=True, preprocessed_name=preprocessed_name, context_length=context_length)


def main():
    BertDataset.split_into_context_length(context_length=128)
    BertDataset.split_into_context_length(context_length=512)
    BertDataset.apply_mlm_and_nsp(context_length=128, num_samples=5_625_000, preprocessed_name="800M_tokens")
    BertDataset.apply_mlm_and_nsp(context_length=512, num_samples=156_250, preprocessed_name="800M_tokens")
    dataset_128 = BertDataset.load(preprocessed_name="800M_tokens", context_length=128)
    dataset_512 = BertDataset.load(preprocessed_name="800M_tokens", context_length=512)
    print(dataset_128)
    print(dataset_512)
    print(len(dataset_128) * 128 + len(dataset_512) * 512)


if __name__ == "__main__":
    main()
