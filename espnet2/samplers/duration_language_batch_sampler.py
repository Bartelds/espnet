import logging
from typing import Iterator, Optional, Tuple, Union, List

from typeguard import typechecked

from espnet2.fileio.read_text import read_2columns_text, load_num_sequence_text
from espnet2.samplers.abs_sampler import AbsSampler
import random
from itertools import chain, zip_longest

random.seed(42)

class DurationLanguageBatchSampler(AbsSampler):
    """BatchSampler.

    This class ensures that each batch only contains
    examples from one language/category. It cycles 
    through languages over the course of training. This class
    returns duration-equalized batches

    """

    @typechecked
    def __init__(
        self,
        batch_size: int,
        shape_files: Union[Tuple[str, ...], List[str]],
        drop_last: bool = True,
        utt2category_file: Optional[str] = None,
        duration_batch_length: int = -1
    ):
        # print("utt2category_file", utt2category_file)
        assert batch_size > 0
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shape_files = shape_files

        # utt2shape:
        #    uttA <anything is o.k>
        #    uttB <anything is o.k>

        utt2shapes = [
            load_num_sequence_text(s, loader_type="csv_int") for s in shape_files
        ]
        
        first_utt2shape = utt2shapes[0]
        for s, d in zip(shape_files, utt2shapes):
            if set(d) != set(first_utt2shape):
                raise RuntimeError(
                    f"keys are mismatched between {s} != {shape_files[0]}"
                )
        
        # first_utt2shape maps files to their audio length
        keys = list(first_utt2shape.keys())

        category2utt = {}
        if utt2category_file is not None:
            utt2category = read_2columns_text(utt2category_file)
            if set(utt2category) != set(keys):
                raise RuntimeError(
                    f"keys are mismatched between {utt2category_file}"
                )
            for k, v in utt2category.items():
                category2utt.setdefault(v, []).append(k)
        else:
            raise Exception(f"utt2category File not Provided!")
        
        # sort utterances by length for each category
        for category in category2utt:
            category2utt[category] = sorted(category2utt[category], key=lambda k:first_utt2shape[k], reverse=True)

        category2len = {}
        for category in category2utt:
            lens = [first_utt2shape[_][0] for _ in category2utt[category]]
            category2len[category] = sum(lens)

        # try and match the batch_size of the longest audio language, like packing
        if duration_batch_length != -1:
            target_dur = duration_batch_length
            print("Target Duration ", target_dur)
        else:
            target_dur = max([
                (category2len[category]/len(category2utt[category]))*self.batch_size for category in category2len])
            print("Target Duration ", target_dur)

        self.category2numbatches = {_:0 for _ in category2len}

        self.batch_list = []

        # implement a greedy solution to the bin-packing problem to generate batches
        category_list = list(category2utt.keys())
        category_list.sort()
        self.category2possiblebatches = [[] for _ in range(len(category_list))]
        for idx, lang in enumerate(category_list):
            self.category2possiblebatches[idx] = [[]]
            bucket_sizes = [0]

            for sample in category2utt[lang]:
                length = first_utt2shape[sample][0]
                found = False
                for i in range(len(bucket_sizes)):
                    if bucket_sizes[i] + length <= target_dur:
                        found = True
                        self.category2possiblebatches[idx][i].append(sample)
                        bucket_sizes[i] += length
                        break

                if not found:
                    self.category2possiblebatches[idx].append([sample])
                    bucket_sizes.append(length)

            self.category2numbatches[lang] = len(bucket_sizes)
            random.shuffle(self.category2possiblebatches[idx])

        # zip ensures that only the first N batches (where N is the minimum number of batches among all languages) are included for each language 
        # self.batch_list = list(chain(*zip(*self.category2possiblebatches)))

        # use this when training with unbalanced datasets (i.e., more than 1 hour per language)
        # Shuffle the resulting batches, trying to ensure batches for each language are distributed as uniformly as possible
        min_list_length = min([len(lst) for lst in self.category2possiblebatches])
        sum_list_length = sum([len(lst) for lst in self.category2possiblebatches])
        props = [len(lst)//min_list_length for lst in self.category2possiblebatches]

        # self.batch_list = list(chain.from_iterable(zip_longest(*self.category2possiblebatches)))
        # # filter None values introduced as padding by zip_longest
        # self.batch_list = [batch for batch in self.batch_list if batch is not None]

        self.batch_list = []
        idxs = [0] * len(self.category2possiblebatches)
        while (len(self.batch_list) < sum_list_length):
            for idx in range(len(self.category2possiblebatches)):
                for _ in range(props[idx]):
                    if (idxs[idx] >= len(self.category2possiblebatches[idx])):
                        continue

                    self.batch_list.append(self.category2possiblebatches[idx][idxs[idx]])
                    idxs[idx] += 1

        chunk_size = sum(props)
        chunks = [self.batch_list[i:min(i+chunk_size, len(self.batch_list))] for i in range(0,len(self.batch_list),chunk_size)]
        for chunk in chunks:
            random.shuffle(chunk)

        # random.shuffle(self.batch_list)

        self.batch_list = [item for chunk in chunks for item in chunk]
        
        # print(self.batch_list)
        # required to init Group DRO
        # self.category2numbatches = category2numbatches
        # print(f"self.num_batches={[(_, len(self.category2possiblebatches[idx])) for idx, _ in enumerate(category_list)]}")
        # print(f"self.batch_list={self.batch_list}")

    def debug_prints(self):
        print(f"batch_size={self.batch_size}")
        print(f"self.batch_list={self.batch_list}")

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"N-batch={len(self)}, "
            f"batch_size={self.batch_size}, "
        )

    def __len__(self):
        return len(self.batch_list)

    def __iter__(self) -> Iterator[Tuple[str, ...]]:
        return iter(self.batch_list)