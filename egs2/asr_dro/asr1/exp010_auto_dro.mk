.ONESHELL:

include cluster_info.mk
EXPERIMENT_ID=exp_010
DATA_SUBSET=1h
USER_SCTK_INSTALL_DIR=
SPECIFIC_LANGUAGES=true
SELECTED_LANGUAGES=pol,deu,myv,swa,cym,ita,nld,slk,kir,urd,asm,lao,hrv,ben,fra
DATASETS=M-AILABS,M-AILABS,commonvoice,commonvoice,commonvoice,commonvoice,commonvoice,commonvoice,commonvoice,commonvoice,commonvoice,fleurs,fleurs,fleurs,fleurs

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
	--exp_dir $(EXP_DIR)/asr_train_$(subst eval_,,$@)/decode_asr_asr_model_valid.loss.ave/test_1h_lid/score_cer/few_shot/trained/ 

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

MMS_LOSS_CTC_0.01_SCEB_GROUPINIT_ARGS= --asr_config conf/$(EXPERIMENT_ID)/train_asr_mms_sceb_dro_0.01_groupinit.yaml

MMS_LOSS_CTC_0.01_ALEB_ARGS= --asr_config conf/$(EXPERIMENT_ID)/train_asr_mms_aleb_dro_0.01.yaml

XLSR_LOSS_CTC_0.01_SCEB_GROUPINIT_ARGS= --asr_config conf/$(EXPERIMENT_ID)/train_asr_xlsr_sceb_dro_0.01_groupinit.yaml

XLSR_LOSS_CTC_0.01_ALEB_ARGS= --asr_config conf/$(EXPERIMENT_ID)/train_asr_xlsr_aleb_dro_0.01.yaml

MMS_LOSS_CTC_0.1_SCEB_GROUPINIT_ARGS= --asr_config conf/$(EXPERIMENT_ID)/train_asr_mms_sceb_dro_0.1_groupinit.yaml

MMS_LOSS_CTC_0.1_ALEB_ARGS= --asr_config conf/$(EXPERIMENT_ID)/train_asr_mms_aleb_dro_0.1.yaml

XLSR_LOSS_CTC_0.1_SCEB_GROUPINIT_ARGS= --asr_config conf/$(EXPERIMENT_ID)/train_asr_xlsr_sceb_dro_0.1_groupinit.yaml

XLSR_LOSS_CTC_0.1_ALEB_ARGS= --asr_config conf/$(EXPERIMENT_ID)/train_asr_xlsr_aleb_dro_0.1.yaml

train_asr_mms_sceb_dro_0.01_groupinit:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(MMS_LOSS_CTC_0.01_SCEB_GROUPINIT_ARGS) $(SCEB_PARAMS)

train_asr_mms_aleb_dro_0.01:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(MMS_LOSS_CTC_0.01_ALEB_ARGS) $(ALEB_PARAMS)

train_asr_xlsr_sceb_dro_0.01_groupinit:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(XLSR_LOSS_CTC_0.01_SCEB_GROUPINIT_ARGS) $(SCEB_PARAMS)

train_asr_xlsr_aleb_dro_0.01:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(XLSR_LOSS_CTC_0.01_ALEB_ARGS) $(ALEB_PARAMS)

train_asr_mms_sceb_dro_0.1_groupinit:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(MMS_LOSS_CTC_0.1_SCEB_GROUPINIT_ARGS) $(SCEB_PARAMS)

train_asr_mms_aleb_dro_0.1:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(MMS_LOSS_CTC_0.1_ALEB_ARGS) $(ALEB_PARAMS)

train_asr_xlsr_sceb_dro_0.1_groupinit:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(XLSR_LOSS_CTC_0.1_SCEB_GROUPINIT_ARGS) $(SCEB_PARAMS)

train_asr_xlsr_aleb_dro_0.1:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(XLSR_LOSS_CTC_0.1_ALEB_ARGS) $(ALEB_PARAMS)

eval_asr_mms_sceb_dro_0.01_groupinit: results/$(EXPERIMENT_ID)/
	$(EVAL_CMD)

eval_asr_mms_aleb_dro_0.01: results/$(EXPERIMENT_ID)/
	$(EVAL_CMD)

eval_asr_xlsr_sceb_dro_0.01_groupinit: results/$(EXPERIMENT_ID)/
	$(EVAL_CMD)

eval_asr_xlsr_aleb_dro_0.01: results/$(EXPERIMENT_ID)/
	$(EVAL_CMD)

eval_asr_mms_sceb_dro_0.1_groupinit: results/$(EXPERIMENT_ID)/
	$(EVAL_CMD)

eval_asr_mms_aleb_dro_0.1: results/$(EXPERIMENT_ID)/
	$(EVAL_CMD)

eval_asr_xlsr_sceb_dro_0.1_groupinit: results/$(EXPERIMENT_ID)/
	$(EVAL_CMD)

eval_asr_xlsr_aleb_dro_0.1: results/$(EXPERIMENT_ID)/
	$(EVAL_CMD)

eval-all: /
	make eval_asr_mms_sceb_dro_0.01_groupinit /
	make eval_asr_mms_aleb_dro_0.01 /
	make eval_asr_xlsr_sceb_dro_0.01_groupinit /
	make eval_asr_xlsr_aleb_dro_0.01 /
	make eval_asr_mms_sceb_dro_0.1_groupinit /
	make eval_asr_mms_aleb_dro_0.1 /
	make eval_asr_xlsr_sceb_dro_0.1_groupinit /
	make eval_asr_xlsr_aleb_dro_0.1 /
	echo 'All done'

