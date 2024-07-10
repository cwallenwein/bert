from data.context_length_splitter import ContextLengthSplitter
from data.nsp_data_processor import NextSentencePredictionDataProcessor
from data.mlm_data_processor import MaskedLanguageModelingDataProcessor
from data.util import get_dataset, get_tokenizer, save_dataset


class BertDataset:

    @staticmethod
    def prepare(context_length: int):
        try:
            get_dataset(split_by_context_length=True, context_length=context_length)
            return True
        except FileNotFoundError or ValueError:
            try:
                dataset = get_dataset(raw=True)
                tokenizer = get_tokenizer()
                context_length_splitter = ContextLengthSplitter(tokenizer=tokenizer)
                dataset = context_length_splitter(
                    dataset=dataset,
                    context_length=context_length
                )
                save_dataset(dataset, split_by_context_length=True, context_length=context_length)
            except Exception as e:
                print(f"Unexpected error: {str(e)}")
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
        return False

    @staticmethod
    def load(num_samples: int, context_length: int):
        tokenizer = get_tokenizer()
        try:
            dataset = get_dataset(split_by_context_length=True, context_length=context_length)
        except FileNotFoundError or ValueError:
            raise Exception(f"Please run BertDataset.prepare(context_length={context_length}) first.")

        nsp_data_collator = NextSentencePredictionDataProcessor(
            tokenizer=tokenizer
        )
        dataset = nsp_data_collator(
            dataset=dataset,
            num_samples=num_samples,
            context_length=context_length
        )

        mlm_data_collator = MaskedLanguageModelingDataProcessor(
            tokenizer=tokenizer,
        )

        dataset = mlm_data_collator(
            dataset=dataset
        )
        assert len(dataset) == num_samples
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
