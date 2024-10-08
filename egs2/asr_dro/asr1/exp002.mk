.ONESHELL:

include cluster_info.mk
EXPERIMENT_ID=exp_002
DATA_SUBSET=1h
DUMP_DIR=outputs/002/dump
EXP_DIR=outputs/002/exp_subset
ASR_STATS_DIR=outputs/002/exp_subset
USER_SCTK_INSTALL_DIR=

COMMON_ARGS=\
	--duration $(DATA_SUBSET) \
	--lid true \
	--only_lid false \
	--dumpdir $(DUMP_DIR) \
	--expdir $(EXP_DIR) \
	--asr_stats_dir $(ASR_STATS_DIR) \

COMMON_TRAIN_ARGS=\
	$(COMMON_ARGS) \
	--stage 11 \
	--asr_tag $@

COMMON_EVAL_ARGS=\
	--exp_dir outputs/exp_subset/asr_train-$(subst eval-,,$@)/decode_asr_asr_model_valid.loss.best/test_1h_lid/score_cer/few_shot/trained/ 

EVAL_CMD=\
	./local/score_macro.sh $(COMMON_EVAL_ARGS) > results/$(EXPERIMENT_ID)/$@.txt

##
# Loss Functions
###
LOSS_CTC_ARGS=\
	--asr_config conf/$(EXPERIMENT_ID)/train_asr_xlsr.yaml

LOSS_CTC_DRO_ARGS=\
	--asr_config conf/$(EXPERIMENT_ID)/train_asr_xlsr_dro.yaml

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
	--asr_config conf/$(EXPERIMENT_ID)/train_asr_xlsr.yaml

preprocess:
	./run_multi.sh \
		$(COMMON_ARGS) \
		$(PREPROCESS_ARGS) \
		--stop_stage 10

preprocess-groups:
	python scripts/dro_scripts/create_groups.py  \
		--utt2spk-file $(DUMP_DIR)/raw/dev_$(DATA_SUBSET)_lid/utt2spk \
		--out-utt2category-file $(DUMP_DIR)/raw/dev_$(DATA_SUBSET)_lid/utt2category 

	python scripts/dro_scripts/create_groups.py  \
		--utt2spk-file $(DUMP_DIR)/raw/train_$(DATA_SUBSET)_lid/utt2spk \
		--out-utt2category-file $(DUMP_DIR)/raw/train_$(DATA_SUBSET)_lid/utt2category 

	python scripts/dro_scripts/create_groups.py  \
		--utt2spk-file $(DUMP_DIR)/raw/test_$(DATA_SUBSET)_lid/utt2spk \
		--out-utt2category-file $(DUMP_DIR)/raw/test_$(DATA_SUBSET)_lid/utt2category 



##
# Training for 6 experimental conditions
###
train-xslr-ctc-aleb:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(LOSS_CTC_ARGS) $(ALEB_PARAMS)

train-xslr-ctc-dro-aleb:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(LOSS_CTC_DRO_ARGS) $(ALEB_PARAMS)

train-xslr-ctc-sceb:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(LOSS_CTC_ARGS) $(SCEB_PARAMS)

train-xslr-ctc-dro-sceb:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(LOSS_CTC_DRO_ARGS) $(SCEB_PARAMS)



##
# Evaluation for 10 experimental conditions
###
results/$(EXPERIMENT_ID)/:
	mkdir -p results/$(EXPERIMENT_ID)/

eval-xslr-ctc-aleb: results/$(EXPERIMENT_ID)/
	$(EVAL_CMD)

eval-xslr-ctc-dro-aleb: results/$(EXPERIMENT_ID)/
	$(EVAL_CMD)

eval-xslr-ctc-sceb: results/$(EXPERIMENT_ID)/
	$(EVAL_CMD)

eval-xslr-ctc-dro-sceb: results/$(EXPERIMENT_ID)/
	$(EVAL_CMD)




eval-all: \
	eval-xslr-ctc-aleb \
	eval-xslr-ctc-dro-aleb \
	eval-xslr-ctc-sceb \
	eval-xslr-ctc-dro-sceb \
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