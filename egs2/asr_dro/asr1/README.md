# CTC-DRO: Robust Optimization for Reducing Language Disparities in Speech Recognition
Code associated with the paper: CTC-DRO: Robust Optimization for Reducing Language Disparities in Speech Recognition.

**Abstract:** Modern deep learning models often achieve high overall performance, but consistently fail on specific subgroups. Group distributionally robust optimization (group DRO) addresses this problem by minimizing the worst-group loss, but it fails when group losses misrepresent performance differences between groups. This is common in domains like speech, where the widely used connectionist temporal classification (CTC) loss scales with input length and varies with linguistic and acoustic properties, leading to spurious differences between group losses. We present CTC-DRO, which addresses the shortcomings of the group DRO objective by smoothing the group weight update to prevent overemphasis on consistently high-loss groups, while using input length-matched batching to mitigate CTC's scaling issues. We evaluate CTC-DRO on the task of multilingual automatic speech recognition (ASR) across five language sets from the ML-SUPERB 2.0 benchmark. CTC-DRO consistently outperforms group DRO and CTC-based baseline models, reducing the worst-language error by up to 65.9% and the average error by up to 47.7%. CTC-DRO can be applied to ASR with minimal computational costs, and offers the potential for reducing group disparities in other domains with similar challenges.

---

## Citation

```bibtex
@misc{bartelds2025ctcdrorobustoptimizationreducing,
      title={CTC-DRO: Robust Optimization for Reducing Language Disparities in Speech Recognition}, 
      author={Martijn Bartelds and Ananjan Nandi and Moussa Koulako Bala Doumbouya and Dan Jurafsky and Tatsunori Hashimoto and Karen Livescu},
      year={2025},
      eprint={2502.01777},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.01777}, 
}
```

---

## Requirements

- Python 3.7+
- PyTorch 1.10+
- ESPnet
- Transformers (for pre-trained models)
- SCTK (for scoring)

See `requirements.txt` for the full list of dependencies.

---

## Installation

```bash
git clone https://github.com/Bartelds/espnet.git
cd egs2/asr_dro/asr1
pip install -r requirements.txt
```

Note: Ensure that your [ESPnet installation](https://espnet.github.io/espnet/installation.html) is correctly configured before proceeding.

---

## Dataset

This repository uses the [ML-SUPERB 2.0 dataset](https://github.com/espnet/espnet/tree/master/egs2/ml_superb/asr1).

After downloading and extracting the dataset, update the dataset path in `db.sh`.

---

### Configuration

Configuration files for model training and inference are located in the `conf/` directory. For example, `mms_example.yaml` contains the CTC-DRO related settings:
```
ctc_conf:
    ctc_type: droctc
    dro_group_count: 6
    dro_step_size: 0.01
    dro_q_epsilon: 1e-10
    init_strategy: uniform
    max_epoch: 40
    num_iters_per_epoch: 1200
    laplace_smoothing: 0.1
```

Other training hyperparameters (e.g., `accum_grad`, `batch_size`, `encoder_conf`, `optim_conf`, etc.) are also specified in these configuration files.

---

## Running experiments

Experiments are controlled via Makefiles. An example can be found in `example.mk`.

Below are example commands for pre-processing data, and training and evaluating models.

### Pre-processing
```bash
make preprocess
make preprocess-groups
```

### Training

Supported hyperparameter options include:
- Step sizes: `0.001`, `0.0001`
- Smoothing values: `0.1`, `0.5`, `1.0`

To train MMS or XLS-R models with a given step-size and smoothing term, use:
```bash
make train_asr_<model>_aleb_dro_<step-size>_la_<smoothing>
```

For example, to train models with a step-size of `0.001` and a smoothing term of `0.1`:
```bash
make train_asr_mms_aleb_dro_0.001_la_0.1
make train_asr_xlsr_aleb_dro_0.001_la_0.1
```

Evaluation commands follow the same pattern:
```bash
make eval_asr_<model>_aleb_dro_<step-size>_la_<smoothing>
```

For example, to evaluate models with a step-size of `0.001` and a smoothing term of `0.1`:
```bash
make eval_asr_mms_aleb_dro_0.001_la_0.1
make eval_asr_xlsr_aleb_dro_0.001_la_0.1
```

Evaluation results will be saved in the `results/EXPERIMENT_ID/` directory.

### Customization

You can customize the languages for training and evaluation by modifying the `SELECTED_LANGUAGES` and `DATASETS` variables in the Makefile:
```
SELECTED_LANGUAGES=pol,spa,ces,ron,nan,cmn
DATASETS=M-AILABS,voxforge,commonvoice,fleurs,commonvoice,fleurs
```
Modify the experiment settings in the Makefile:
```
EXPERIMENT_ID=exp_000  # Experiment identifier
DATA_SUBSET=1h         # Data duration (10min or 1h)
```

---

## Repository structure
```
├── conf/                       # YAML configuration files for training and inference (including CTC-DRO parameters)
├── data/                       # Data preparation scripts and dataset directories
├── dump/                       # Model dump directories
├── exp/                        # Experiment outputs and logs
├── local/                      # Local helper scripts (e.g., scoring)
├── scripts/                    # Additional helper scripts
├── Makefile                    # Top-level Makefile for experiment management
├── example.mk                  # Experiment configuration (customize as needed)
├── requirements.txt            # Python dependencies
├── run_multi.sh                # Script that wraps the training pipeline
├── db.sh                       # Dataset path configuration file
└── README.md                   # Main README file
```
