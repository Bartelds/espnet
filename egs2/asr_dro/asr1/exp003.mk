.ONESHELL:

include cluster_info.mk
EXPERIMENT_ID=exp_003
DATA_SUBSET=1h
USER_SCTK_INSTALL_DIR=
SPECIFIC_LANGUAGES=true
SELECTED_LANGUAGES=hrv,afr,mkd,fin,eng,spa,sah,nso,nld,deu
DATASETS=fleurs,nchlt,fleurs,fleurs,LAD,mls,commonvoice,nchlt,commonvoice,voxforge

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
	--exp_dir $(EXP_DIR)/asr_train-$(subst eval-,,$@)/decode_asr_asr_model_valid.loss.best/test_1h_lid/score_cer/few_shot/trained/ 

EVAL_CMD=\
	./local/score_macro.sh $(COMMON_EVAL_ARGS) > results/$(EXPERIMENT_ID)/$@.txt

##
# Loss Functions
###
# Hparam sweeps can be done here
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
train-xlsr-ctc-aleb:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(LOSS_CTC_ARGS) $(ALEB_PARAMS)

train-xlsr-ctc-dro-aleb:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(LOSS_CTC_DRO_ARGS) $(ALEB_PARAMS)

train-xlsr-ctc-sceb:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(LOSS_CTC_ARGS) $(SCEB_PARAMS)

train-xlsr-ctc-dro-sceb:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(LOSS_CTC_DRO_ARGS) $(SCEB_PARAMS)



##
# Evaluation for 10 experimental conditions
###
results/$(EXPERIMENT_ID)/:
	mkdir -p results/$(EXPERIMENT_ID)/

eval-xlsr-ctc-aleb: results/$(EXPERIMENT_ID)/
	$(EVAL_CMD)

eval-xlsr-ctc-dro-aleb: results/$(EXPERIMENT_ID)/
	$(EVAL_CMD)

eval-xlsr-ctc-sceb: results/$(EXPERIMENT_ID)/
	$(EVAL_CMD)

eval-xlsr-ctc-dro-sceb: results/$(EXPERIMENT_ID)/
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