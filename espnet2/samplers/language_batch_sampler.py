import logging
from typing import Iterator, Optional, Tuple

from typeguard import typechecked

from espnet2.fileio.read_text import read_2columns_text
from espnet2.samplers.abs_sampler import AbsSampler
import random

random.seed(42)

class LanguageBatchSampler(AbsSampler):
    """BatchSampler with constant batch-size.

    This class ensures that each batch only contains
    examples from one language/category. It cycles 
    through languages over the course of training.

    Args:
        batch_size:
        key_file:
    """

    @typechecked
    def __init__(
        self,
        batch_size: int,
        key_file: str,
        drop_last: bool = True,
        utt2category_file: Optional[str] = None,
    ):
        print("utt2category_file", utt2category_file)
        assert batch_size > 0
        self.batch_size = batch_size
        self.key_file = key_file
        self.drop_last = drop_last

        # utt2shape:
        #    uttA <anything is o.k>
        #    uttB <anything is o.k>
        utt2any = read_2columns_text(key_file)
        if len(utt2any) == 0:
            logging.warning(f"{key_file} is empty")
        # In this case the, the first column in only used
        keys = list(utt2any)
        if len(keys) == 0:
            raise RuntimeError(f"0 lines found: {key_file}")

        category2utt = {}
        if utt2category_file is not None:
            utt2category = read_2columns_text(utt2category_file)
            if set(utt2category) != set(keys):
                raise RuntimeError(
                    f"keys are mismatched between {utt2category_file} != {key_file}"
                )
            for k, v in utt2category.items():
                category2utt.setdefault(v, []).append(k)
        else:
            raise Exception(f"utt2category File not Provided!")

        self.batch_list = []
        # Maintain iterators for all categories
        # Pick random language and create batch until all iterators hit their end

        def check_complete(it):
            return sum(it.values()) == 0

        def get_nonempty(it, s):
            # get next index for which iterator value is not 0
            it_keys = list(it.keys())
            if it[it_keys[s]] != 0:
                return s
            k = s
            while it[it_keys[k]] == 0:
                k = (k+1)%len(it)
                if k == s:
                    return s
            return k

        cat_iterators = {d:len(category2utt[d]) for d in category2utt}
        cats = list(category2utt.keys())
        while not check_complete(cat_iterators):
            lang_index = get_nonempty(cat_iterators, random.randint(0, len(cats) - 1))
            lang = cats[get_nonempty(cat_iterators, lang_index)]
            category_keys = category2utt[lang]
            # Apply max(, 1) to avoid 0-batches
            if self.drop_last and cat_iterators[lang] < batch_size:
                    # drop incomplete batches
                    cat_iterators[lang] = 0
            else:
                # it at end -> selecting entries from the start
                start = len(category_keys) - cat_iterators[lang]
                decrement = min(batch_size, cat_iterators[lang])
                end = start + decrement
                cat_iterators[lang] -= decrement

                curr_ex = category_keys[start:end]
                if decrement < batch_size:
                    # pad out batch
                    curr_ex += [category_keys[start] for _ in range(batch_size - decrement)]

                self.batch_list.append(curr_ex)

    def debug_prints(self):
        print(f"batch_size={self.batch_size}")
        print(f"self.batch_list={self.batch_list[:20]}")

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"N-batch={len(self)}, "
            f"batch_size={self.batch_size}, "
            f"key_file={self.key_file}, "
        )

    def __len__(self):
        return len(self.batch_list)

    def __iter__(self) -> Iterator[Tuple[str, ...]]:
        return iter(self.batch_list)
