.ONESHELL:

include cluster_info.mk
EXPERIMENT_ID=exp_022
DATA_SUBSET=1h
USER_SCTK_INSTALL_DIR=
SPECIFIC_LANGUAGES=true
SELECTED_LANGUAGES=lao
DATASETS=fleurs

DUMP_DIR=$(DUMP_DIR_BASE)_$(EXPERIMENT_ID)
EXP_DIR=$(EXP_DIR_BASE)_$(EXPERIMENT_ID)
ASR_STATS_DIR=$(ASR_STATS_DIR_BASE)_$(EXPERIMENT_ID)

COMMON_ARGS=\
	--duration $(DATA_SUBSET) \
	--lid true \
	--only_lid false \
	--dumpdir $(DUMP_DIR) \
	--expdir $(EXP_DIR) \
	--asr_stats_dir $(ASR_STATS_DIR) \
	--specific_lang $(SPECIFIC_LANGUAGES) \
	--selected_languages $(SELECTED_LANGUAGES) \
	--datasets $(DATASETS)

COMMON_TRAIN_ARGS=\
	$(COMMON_ARGS) \
	--stage 11 \
	--asr_tag $@

COMMON_EVAL_ARGS=\
	--exp_dir exp/_$(EXP_DIR)/asr_train-$(subst eval-,,$@)/decode_asr_asr_model_valid.loss.best/test_1h_lid/score_cer/few_shot/trained/ 

EVAL_CMD=\
	./local/score_macro.sh $(COMMON_EVAL_ARGS) > results/$(EXPERIMENT_ID)/$@.txt

##
# Loss Functions
###
# Hparam sweeps can be done here

##
# Batch Sampling Methods
###
SCEB_PARAMS=\
	--batch_type language 

ALEB_PARAMS=\
	--batch_type duration_language 

##
# Preprocessing
###
PREPROCESS_ARGS=\
	--asr_config conf/$(EXPERIMENT_ID)/train_asr.yaml

preprocess:
	./run_multi.sh \
		$(COMMON_ARGS) \
		$(PREPROCESS_ARGS) \
		--stop_stage 10

preprocess-groups:
	python scripts/dro_scripts/create_groups.py  \
		--utt2spk-file $(DUMP_DIR)/raw/dev_$(DATA_SUBSET)$(SUFFIX)/utt2spk \
		--out-utt2category-file $(DUMP_DIR)/raw/dev_$(DATA_SUBSET)$(SUFFIX)/utt2category 

	python scripts/dro_scripts/create_groups.py  \
		--utt2spk-file $(DUMP_DIR)/raw/train_$(DATA_SUBSET)$(SUFFIX)/utt2spk \
		--out-utt2category-file $(DUMP_DIR)/raw/train_$(DATA_SUBSET)$(SUFFIX)/utt2category 

	python scripts/dro_scripts/create_groups.py  \
		--utt2spk-file $(DUMP_DIR)/raw/test_$(DATA_SUBSET)$(SUFFIX)/utt2spk \
		--out-utt2category-file $(DUMP_DIR)/raw/test_$(DATA_SUBSET)$(SUFFIX)/utt2category 

##
# Training for 6 experimental conditions
###


##
# Evaluation for 10 experimental conditions
###
results/$(EXPERIMENT_ID)/:
	mkdir -p results/$(EXPERIMENT_ID)/

eval-all: \
	eval-xlsr-ctc-aleb \
	eval-xlsr-ctc-dro-aleb \
	eval-xlsr-ctc-sceb \
	eval-xlsr-ctc-dro-sceb \
	echo "done"

##
# Cluster Management Tools
###
submit-target-to-cluster:
	echo "#!/bin/bash" > slurm_job.sh
	echo "cd `pwd`" >> slurm_job.sh
	echo "source ../../../tools/activate_python.sh" >> slurm_job.sh
	echo "make $(TARGET)" >> slurm_job.sh
	sbatch \
			--time=120:00:00 \
			--cpus-per-task=64 \
			--gres=gpu:1 \
			--mem 64G \
			--account $(SLURM_CLUSTER_ACCOUNT_NAME) \
			--partition $(SLURM_CLUSTER_PARTITION_NAME) \
			--job-name $(TARGET) \
			--exclude=$(SLURM_CLUSTER_EXCLUDE_NODES) \
			slurm_job.sh 

show-jobs:
	squeue --format="%.18i %.12P %70j %.8T %.10M %.9l %.6D %R" -u moussa

activate-venv:
	source ../../../tools/activate_python.sh 

MMS_LOSS_CTC_0.0005_ARGS= --asr_config conf/$(EXPERIMENT_ID)/train_asr_mms_0.0005.yaml

XLSR_LOSS_CTC_0.0005_ARGS= --asr_config conf/$(EXPERIMENT_ID)/train_asr_xlsr_0.0005.yaml

MMS_LOSS_CTC_0.0001_ARGS= --asr_config conf/$(EXPERIMENT_ID)/train_asr_mms_0.0001.yaml

XLSR_LOSS_CTC_0.0001_ARGS= --asr_config conf/$(EXPERIMENT_ID)/train_asr_xlsr_0.0001.yaml

MMS_LOSS_CTC_5e-05_ARGS= --asr_config conf/$(EXPERIMENT_ID)/train_asr_mms_5e-05.yaml

XLSR_LOSS_CTC_5e-05_ARGS= --asr_config conf/$(EXPERIMENT_ID)/train_asr_xlsr_5e-05.yaml

MMS_LOSS_CTC_1e-05_ARGS= --asr_config conf/$(EXPERIMENT_ID)/train_asr_mms_1e-05.yaml

XLSR_LOSS_CTC_1e-05_ARGS= --asr_config conf/$(EXPERIMENT_ID)/train_asr_xlsr_1e-05.yaml

train-mms-ctc-aleb-0.0005:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(MMS_LOSS_CTC_0.0005_ARGS) $(ALEB_PARAMS)

train-xlsr-ctc-aleb-0.0005:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(XLSR_LOSS_CTC_0.0005_ARGS) $(ALEB_PARAMS)

train-mms-ctc-sceb-0.0005:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(MMS_LOSS_CTC_0.0005_ARGS) $(SCEB_PARAMS)

train-xlsr-ctc-sceb-0.0005:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(XLSR_LOSS_CTC_0.0005_ARGS) $(SCEB_PARAMS)

train-mms-ctc-aleb-0.0001:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(MMS_LOSS_CTC_0.0001_ARGS) $(ALEB_PARAMS)

train-xlsr-ctc-aleb-0.0001:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(XLSR_LOSS_CTC_0.0001_ARGS) $(ALEB_PARAMS)

train-mms-ctc-sceb-0.0001:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(MMS_LOSS_CTC_0.0001_ARGS) $(SCEB_PARAMS)

train-xlsr-ctc-sceb-0.0001:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(XLSR_LOSS_CTC_0.0001_ARGS) $(SCEB_PARAMS)

train-mms-ctc-aleb-5e-05:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(MMS_LOSS_CTC_5e-05_ARGS) $(ALEB_PARAMS)

train-xlsr-ctc-aleb-5e-05:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(XLSR_LOSS_CTC_5e-05_ARGS) $(ALEB_PARAMS)

train-mms-ctc-sceb-5e-05:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(MMS_LOSS_CTC_5e-05_ARGS) $(SCEB_PARAMS)

train-xlsr-ctc-sceb-5e-05:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(XLSR_LOSS_CTC_5e-05_ARGS) $(SCEB_PARAMS)

train-mms-ctc-aleb-1e-05:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(MMS_LOSS_CTC_1e-05_ARGS) $(ALEB_PARAMS)

train-xlsr-ctc-aleb-1e-05:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(XLSR_LOSS_CTC_1e-05_ARGS) $(ALEB_PARAMS)

train-mms-ctc-sceb-1e-05:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(MMS_LOSS_CTC_1e-05_ARGS) $(SCEB_PARAMS)

train-xlsr-ctc-sceb-1e-05:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(XLSR_LOSS_CTC_1e-05_ARGS) $(SCEB_PARAMS)

eval-mms-ctc-aleb-0.0005: results/$(EXPERIMENT_ID)/$(EVAL_CMD)

eval-xlsr-ctc-aleb-0.0005: results/$(EXPERIMENT_ID)/$(EVAL_CMD)

eval-mms-ctc-sceb-0.0005: results/$(EXPERIMENT_ID)/$(EVAL_CMD)

eval-xlsr-ctc-sceb-0.0005: results/$(EXPERIMENT_ID)/$(EVAL_CMD)

eval-mms-ctc-aleb-0.0001: results/$(EXPERIMENT_ID)/$(EVAL_CMD)

eval-xlsr-ctc-aleb-0.0001: results/$(EXPERIMENT_ID)/$(EVAL_CMD)

eval-mms-ctc-sceb-0.0001: results/$(EXPERIMENT_ID)/$(EVAL_CMD)

eval-xlsr-ctc-sceb-0.0001: results/$(EXPERIMENT_ID)/$(EVAL_CMD)

eval-mms-ctc-aleb-5e-05: results/$(EXPERIMENT_ID)/$(EVAL_CMD)

eval-xlsr-ctc-aleb-5e-05: results/$(EXPERIMENT_ID)/$(EVAL_CMD)

eval-mms-ctc-sceb-5e-05: results/$(EXPERIMENT_ID)/$(EVAL_CMD)

eval-xlsr-ctc-sceb-5e-05: results/$(EXPERIMENT_ID)/$(EVAL_CMD)

eval-mms-ctc-aleb-1e-05: results/$(EXPERIMENT_ID)/$(EVAL_CMD)

eval-xlsr-ctc-aleb-1e-05: results/$(EXPERIMENT_ID)/$(EVAL_CMD)

eval-mms-ctc-sceb-1e-05: results/$(EXPERIMENT_ID)/$(EVAL_CMD)

eval-xlsr-ctc-sceb-1e-05: results/$(EXPERIMENT_ID)/$(EVAL_CMD)

