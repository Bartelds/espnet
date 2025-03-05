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

Finally, run
```bash
source ../../../tools/activate_python.sh
```

---

## Dataset

This repository uses the [ML-SUPERB 2.0 dataset](https://github.com/espnet/espnet/tree/master/egs2/ml_superb/asr1).

After downloading and extracting the dataset, update the dataset path in `db.sh`.

---

### Configuration

Configuration files for model training and inference are located in the `conf/` directory. We expect configs for different language subsets and settings to be organized in separate folders, for example, the configs for Experiment 1 may be stored inside `conf/exp_001/`. We provide example configs for a non-DRO baseline (`mms_example_baseline.yaml` and `xlsr_example_baseline.yaml`), Group DRO (`mms_example_group_dro.yaml` and `xlsr_example_group_dro.yaml`) and CTC-DRO (`mms_example_ctc_dro.yaml` and `xlsr_example_ctc_dro.yaml`) for both XLS-R and MMS models. Please also copy `train_asr.yaml` inside this folder, it contains the config for preprocessing files.

The CTC-DRO related settings are as follows:
```
ctc_conf:
    accumulation: true
    agg: sum
    ctc_type: droctc
    dro_group_count: 6
    dro_q_epsilon: 1.0e-10
    dro_step_size: 0.0001
    smoothing: 0.1
    normalize_grad: true
```

On the other hand, a Group DRO config looks like
```
ctc_conf:
    accumulation: false
    agg: mean
    ctc_type: droctc
    dro_group_count: 6
    dro_q_epsilon: 1.0e-10
    dro_step_size: 0.0001
    smoothing: 0.0
    normalize_grad: false
```


Other training hyperparameters (e.g., `accum_grad`, `batch_size`, `encoder_conf`, `optim_conf`, etc.) are also specified in these configuration files.

Config files can also automatically be created for hyperparameter sweeps by changing the global variables at the top of `lr_sweep_baseline.py`, `lr_sweep_group_dro.py` and `lr_sweep_ctc_dro.py` and running these files.

Supported hyperparameter options for CTC-DRO include the step size and smoothing hyperparameter. The learning rate for the baseline and step size for Group DRO can also be swept over using these files.

---

## Running experiments

Experiments are controlled via Makefiles. 
Before running any experiments, please populate `cluster_info.mk` with the `DUMP_DIR_BASE` (location of preprocessed data files, example: `scr/dump`), `EXP_DIR_BASE` (location to save models, example: `scr/exp`) and `ASR_STATS_DIR_BASE` (location containing statistics for the dataset, typically the same as `EXP_DIR_BASE`, example: `scr/exp`).

An example Makefile can be found in `exp001_m.mk`. To run experiments without DRO, `create_makefile_baseline.py` can be run after setting appropriate hyperparameters at the top of the file, which creates the Makefile `exp001_auto_baseline.mk`. Similarly, for experiments with Group DRO, `create_makefile_group_dro.py` creates `exp001_auto_group_dro.mk` and for experiments with CTC-DRO, `create_makefile_ctc_dro.py` creates `exp001_auto_ctc_dro.mk`. These files can then be included inside `Makefile` to run experiments.

The commands for pre-processing data before training are:

### Pre-processing
```bash
make preprocess
make preprocess-groups
```

### Training

Supported hyperparameter options include:
- Step sizes: `0.001`, `0.0001`
- Smoothing values: `0.1`, `0.5`, `1.0`

To train MMS or XLS-R models with CTC-DRO, a given step-size and smoothing term, use:
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

To train MMS or XLS-R models with Group DRO, and a given step-size, use:
```bash
make train_asr_<model>_aleb_dro_<step-size>_base
```

To evaluate, 
```bash
make eval_asr_<model>_aleb_dro_<step-size>_base
```

To train MMS or XLS-R models without DRO, and a given learning rate, use:
```bash
make train_<model>_ctc_aleb_<learning_rate>
```

To evaluate, 
```bash
make eval_<model>_ctc_aleb_<learning_rate>
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
EXPERIMENT_ID=exp_001  # Experiment identifier
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
