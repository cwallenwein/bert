from data.context_length_splitter import ContextLengthSplitter
from data.nsp_data_processor import NextSentencePredictionDataProcessor
from data.mlm_data_processor import MaskedLanguageModelingDataProcessor
from data.util import get_dataset, get_tokenizer, save_dataset
from datasets import enable_progress_bars, disable_progress_bars


class BertDataset:
    # TODO: give option that both sentences for nsp are not actual sentences but just spans of contiguous text
    # TODO: compare performance

    @staticmethod
    def prepare(context_length: int, verbose: bool = True):
        try:
            get_dataset(split_by_context_length=True, context_length=context_length)
            return True
        except FileNotFoundError or ValueError:
            try:
                dataset = get_dataset(raw=True)
                tokenizer = get_tokenizer()
                if not verbose:
                    disable_progress_bars()
                context_length_splitter = ContextLengthSplitter(tokenizer=tokenizer)
                dataset = context_length_splitter(
                    dataset=dataset,
                    context_length=context_length
                )
                save_dataset(dataset, split_by_context_length=True, context_length=context_length)
                enable_progress_bars()
                return True
            except Exception as e:
                print(f"Unexpected error: {str(e)}")
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
        enable_progress_bars()
        return False

    @staticmethod
    def load(num_samples: int, context_length: int, verbose: bool = True):
        tokenizer = get_tokenizer()
        try:
            dataset = get_dataset(split_by_context_length=True, context_length=context_length)
        except FileNotFoundError or ValueError:
            raise Exception(f"Please run BertDataset.prepare(context_length={context_length}) first.")

        if not verbose:
            disable_progress_bars()

        nsp_data_collator = NextSentencePredictionDataProcessor(
            tokenizer=tokenizer
        )
        dataset = nsp_data_collator(
            dataset=dataset,
            num_samples=num_samples,
            context_length=context_length,
            verbose=verbose
        )

        mlm_data_collator = MaskedLanguageModelingDataProcessor(
            tokenizer=tokenizer,
        )

        dataset = mlm_data_collator(
            dataset=dataset
        )
        assert len(dataset) == num_samples
        enable_progress_bars()

        return dataset


def main():
    BertDataset.prepare(context_length=128)
    BertDataset.prepare(context_length=512)
    dataset_128 = BertDataset.load(num_samples=100_000, context_length=128)
    dataset_512 = BertDataset.load(num_samples=100_000, context_length=512)
    print(dataset_128)
    print(dataset_512)


if __name__ == "__main__":
    main()
