# CTC-DRO: Robust Optimization for Reducing Language Disparities in Speech Recognition
Code associated with the paper: CTC-DRO: Robust Optimization for Reducing Language Disparities in Speech Recognition.

**Abstract:** Modern deep learning models often achieve high overall performance, but consistently fail on specific subgroups. Group distributionally robust optimization (group DRO) addresses this problem by minimizing the worst-group loss, but it fails when group losses misrepresent performance differences between groups. This is common in domains like speech, where the widely used connectionist temporal classification (CTC) loss scales with input length and varies with linguistic and acoustic properties, leading to spurious differences between group losses. We present CTC-DRO, which addresses the shortcomings of the group DRO objective by smoothing the group weight update to prevent overemphasis on consistently high-loss groups, while using input length-matched batching to mitigate CTC's scaling issues. We evaluate CTC-DRO on the task of multilingual automatic speech recognition (ASR) across five language sets from the ML-SUPERB 2.0 benchmark. CTC-DRO consistently outperforms group DRO and CTC-based baseline models, reducing the worst-language error by up to 47.1% and the average error by up to 32.9%. CTC-DRO can be applied to ASR with minimal computational costs, and offers the potential for reducing group disparities in other domains with similar challenges.

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

- Python 3.7 or higher
- PyTorch 1.10 or higher
- ESPnet
- Transformers (for pre-trained models)
- SCTK (for scoring)

For a complete list of dependencies, please refer to the `requirements.txt` file.

---

## Installation

```bash
git clone https://github.com/Bartelds/espnet.git
cd egs2/asr_dro/asr1
pip install -r requirements.txt
```

Note: Ensure that your [ESPnet installation](https://espnet.github.io/espnet/installation.html) is correctly configured before proceeding.

Finally, activate the Python environment:
```bash
source ../../../tools/activate_python.sh
```

---

## Dataset

This repository uses the [ML-SUPERB 2.0 dataset](https://github.com/espnet/espnet/tree/master/egs2/ml_superb/asr1).

After downloading and extracting the dataset, update the dataset path (i.e., `ML-SUPERB` variable) in `db.sh`.

---

## Configuration

Configuration files for model training and inference are located in the `conf/` directory. We expect that configurations for different language subsets and experimental settings will be organized in separate subfolders. For example, the configuration files for Experiment 1 may be stored in `conf/exp_001/`.

Within the `conf/` directory, you will find example configuration files for the three training approaches:
- **CTC baseline:**  
  - `mms_example_baseline.yaml`  
  - `xlsr_example_baseline.yaml`
- **Group DRO:**  
  - `mms_example_group_dro.yaml`  
  - `xlsr_example_group_dro.yaml`
- **CTC-DRO:**  
  - `mms_example_ctc_dro.yaml`  
  - `xlsr_example_ctc_dro.yaml`

Additionally, please copy the `train_asr.yaml` file into each experiment folder as it contains the configuration for data preprocessing.

Below are example configuration snippets:

**CTC-DRO:**
```yaml
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

**Group DRO:**
```yaml
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

Other training hyperparameters (e.g., `accum_grad`, `batch_size`, `encoder_conf`, `optim_conf`, etc.) are defined within these configuration files. For hyperparameter sweeps, adjust the global variables at the top of `lr_sweep_baseline.py`, `lr_sweep_group_dro.py`, and `lr_sweep_ctc_dro.py`, and then run these scripts to automatically generate new configuration files.

---

## Running experiments

Experiments are controlled via Makefiles. Before running any experiments, populate `cluster_info.mk` with:
- `DUMP_DIR_BASE`: Location of preprocessed data files (e.g., `scr/dump`)
- `EXP_DIR_BASE`: Directory to save models (e.g., `scr/exp`)
- `ASR_STATS_DIR_BASE`: Directory containing dataset statistics (typically the same as `EXP_DIR_BASE`)

For each experiment, generate the appropriate Makefile:
- **CTC baseline:** Run `create_makefile_baseline.py` to create `exp001_auto_baseline.mk`
- **Group DRO:** Run `create_makefile_group_dro.py` to create `exp001_auto_group_dro.mk`
- **CTC-DRO:** Run `create_makefile_ctc_dro.py` to create `exp001_auto_ctc_dro.mk`

An example Makefile is provided as `exp001_m.mk`. Include the generated Makefile in the main `Makefile` to run experiments.

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

To train MMS or XLS-R models with CTC-DRO:
```bash
make train_asr_mms_aleb_dro_<step-size>_la_<smoothing>
make train_asr_xlsr_aleb_dro_<step-size>_la_<smoothing>
```

For example, with a step size of 0.001 and smoothing of 0.1:
```bash
make train_asr_mms_aleb_dro_0.001_la_0.1
make train_asr_xlsr_aleb_dro_0.001_la_0.1
```

To evaluate these models:
```bash
make eval_asr_mms_aleb_dro_0.001_la_0.1
make eval_asr_xlsr_aleb_dro_0.001_la_0.1
```

To train and evaluate models with Group DRO:
```bash
make train_asr_<model>_aleb_dro_<step-size>_base
make eval_asr_<model>_aleb_dro_<step-size>_base
```

To train and evaluate baseline models, specify the chosen learning rate:
```bash
make train_<model>_ctc_aleb_<learning_rate>
make eval_<model>_ctc_aleb_<learning_rate>
```

Evaluation results will be saved in the `results/EXPERIMENT_ID/` directory.

### Customization

You can customize the languages for training and evaluation by modifying the `SELECTED_LANGUAGES` and `DATASETS` variables in the Makefile.
For example:
```makefile
SELECTED_LANGUAGES=pol,spa,ces,ron,nan,cmn
DATASETS=M-AILABS,voxforge,commonvoice,fleurs,commonvoice,fleurs
```
Modify the experiment settings in the Makefile:
```makefile
EXPERIMENT_ID=exp_001  # Experiment identifier
DATA_SUBSET=1h         # Data duration (10min or 1h)
```

---

## Released models

The following table lists the released models. Each model is available for download via Hugging Face.

| Language Set      | Model Type | Method      | Download Link                                                             |
|-------------------|------------|-------------|---------------------------------------------------------------------------|
| Set 1             | MMS        | Baseline    | [Download](https://huggingface.co/bartelds/ctc-baseline_mms_set_1)        |
| Set 1             | MMS        | Group DRO   | [Download](https://huggingface.co/bartelds/group-dro_mms_set_1)           |
| Set 1             | MMS        | CTC-DRO     | [Download](https://huggingface.co/bartelds/ctc-dro_mms_set_1)             |
| Set 1             | XLS-R      | Baseline    | [Download](https://huggingface.co/bartelds/ctc-baseline_xlsr_set_1)       |
| Set 1             | XLS-R      | Group DRO   | [Download](https://huggingface.co/bartelds/group-dro_xlsr_set_1)          |
| Set 1             | XLS-R      | CTC-DRO     | [Download](https://huggingface.co/bartelds/ctc-dro_xlsr_set_1)            |
| Set 2             | MMS        | Baseline    | [Download](https://huggingface.co/bartelds/ctc-baseline_mms_set_2)        |
| Set 2             | MMS        | Group DRO   | [Download](https://huggingface.co/bartelds/group-dro_mms_set_2)           |
| Set 2             | MMS        | CTC-DRO     | [Download](https://huggingface.co/bartelds/ctc-dro_mms_set_2)             |
| Set 2             | XLS-R      | Baseline    | [Download](https://huggingface.co/bartelds/ctc-baseline_xlsr_set_2)       |
| Set 2             | XLS-R      | Group DRO   | [Download](https://huggingface.co/bartelds/group-dro_xlsr_set_2)          |
| Set 2             | XLS-R      | CTC-DRO     | [Download](https://huggingface.co/bartelds/ctc-dro_xlsr_set_2)            |
| Set 3             | MMS        | Baseline    | [Download](https://huggingface.co/bartelds/ctc-baseline_mms_set_3)        |
| Set 3             | MMS        | Group DRO   | [Download](https://huggingface.co/bartelds/group-dro_mms_set_3)           |
| Set 3             | MMS        | CTC-DRO     | [Download](https://huggingface.co/bartelds/ctc-dro_mms_set_3)             |
| Set 3             | XLS-R      | Baseline    | [Download](https://huggingface.co/bartelds/ctc-baseline_xlsr_set_3)       |
| Set 3             | XLS-R      | Group DRO   | [Download](https://huggingface.co/bartelds/group-dro_xlsr_set_3)          |
| Set 3             | XLS-R      | CTC-DRO     | [Download](https://huggingface.co/bartelds/ctc-dro_xlsr_set_3)            |
| Set 4             | MMS        | Baseline    | [Download](https://huggingface.co/bartelds/ctc-baseline_mms_set_4)        |
| Set 4             | MMS        | Group DRO   | [Download](https://huggingface.co/bartelds/group-dro_mms_set_4)           |
| Set 4             | MMS        | CTC-DRO     | [Download](https://huggingface.co/bartelds/ctc-dro_mms_set_4)             |
| Set 4             | XLS-R      | Baseline    | [Download](https://huggingface.co/bartelds/ctc-baseline_xlsr_set_4)       |
| Set 4             | XLS-R      | Group DRO   | [Download](https://huggingface.co/bartelds/group-dro_xlsr_set_4)          |
| Set 4             | XLS-R      | CTC-DRO     | [Download](https://huggingface.co/bartelds/ctc-dro_xlsr_set_4)            |
| Set 5             | MMS        | Baseline    | [Download](https://huggingface.co/bartelds/ctc-baseline_mms_set_5)        |
| Set 5             | MMS        | Group DRO   | [Download](https://huggingface.co/bartelds/group-dro_mms_set_5)           |
| Set 5             | MMS        | CTC-DRO     | [Download](https://huggingface.co/bartelds/ctc-dro_mms_set_5)             |
| Set 5             | XLS-R      | Baseline    | [Download](https://huggingface.co/bartelds/ctc-baseline_xlsr_set_5)       |
| Set 5             | XLS-R      | Group DRO   | [Download](https://huggingface.co/bartelds/group-dro_xlsr_set_5)          |
| Set 5             | XLS-R      | CTC-DRO     | [Download](https://huggingface.co/bartelds/ctc-dro_xlsr_set_5)            |
| Set 1 (all)       | MMS        | Baseline    | [Download](https://huggingface.co/bartelds/ctc-baseline_mms_set_1-extra)  |
| Set 1 (all)       | MMS        | Group DRO   | [Download](https://huggingface.co/bartelds/group-dro_mms_set_1-extra)     |
| Set 1 (all)       | MMS        | CTC-DRO     | [Download](https://huggingface.co/bartelds/ctc-dro_mms_set_1-extra)       |
| Set 1 (all)       | XLS-R      | Baseline    | [Download](https://huggingface.co/bartelds/ctc-baseline_xlsr_set_1-extra) |
| Set 1 (all)       | XLS-R      | Group DRO   | [Download](https://huggingface.co/bartelds/group-dro_xlsr_set_1-extra)    |
| Set 1 (all)       | XLS-R      | CTC-DRO     | [Download](https://huggingface.co/bartelds/ctc-dro_xlsr_set_1-extra)      |
| Set 2 (all)       | MMS        | Baseline    | [Download](https://huggingface.co/bartelds/ctc-baseline_mms_set_2-extra)  |
| Set 2 (all)       | MMS        | Group DRO   | [Download](https://huggingface.co/bartelds/group-dro_mms_set_2-extra)     |
| Set 2 (all)       | MMS        | CTC-DRO     | [Download](https://huggingface.co/bartelds/ctc-dro_mms_set_2-extra)       |
| Set 2 (all)       | XLS-R      | Baseline    | [Download](https://huggingface.co/bartelds/ctc-baseline_xlsr_set_2-extra) |
| Set 2 (all)       | XLS-R      | Group DRO   | [Download](https://huggingface.co/bartelds/group-dro_xlsr_set_2-extra)    |
| Set 2 (all)       | XLS-R      | CTC-DRO     | [Download](https://huggingface.co/bartelds/ctc-dro_xlsr_set_2-extra)      |
