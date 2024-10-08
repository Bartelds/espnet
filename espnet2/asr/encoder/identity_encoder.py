
# Copyright 2023 
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""Identity encoder definition."""

from typing import Optional, Tuple

import torch
from typeguard import typechecked

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from typing import abstractmethod


class IdentityEncoder(AbsEncoder):
    def __init__(
        self,
        input_size: int
    ):
        super().__init__()
        self._output_size = input_size

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        return xs_pad, ilens, prev_states


