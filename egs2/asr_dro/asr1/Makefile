.ONESHELL:

include cluster_info.mk
# SLURM_CLUSTER_ACCOUNT_NAME=cluster-account-name
# SLURM_CLUSTER_PARTITION_NAME=cluster-partition-name
# SLURM_CLUSTER_EXCLUDE_NODES=node1,node2,node3


# train-asr-dro-baseline:
# 	/nlp/scr/moussa/git/asr-dro/espnet/tools/activate_python.sh
# 	cd /nlp/scr/moussa/git/asr-dro/espnet/egs2/asr_dro/asr1
# 	./run_multi.sh --asr_config conf/train_asr_dro.yaml --duration 1h --lid true --stage 5

DATA_SUBSET=1h
# DUMP_DIR=outputs/dump
# EXP_DIR=outputs/exp_subset
# ASR_STATS_DIR=outputs/exp_subset

COMMON_ARGS=\
	--duration $(DATA_SUBSET) \
	--lid true \
	--only_lid false \
	--dumpdir $(DUMP_DIR) \
	--expdir $(EXP_DIR) \
	--asr_stats_dir $(ASR_STATS_DIR) \
	--batch_type language


PREPROCESS_ARGS=\
	--asr_config conf/train_asr.yaml

XLSR_ARGS=\
	--asr_tag train_asr_xlsr_duration \
	--asr_config conf/train_asr_xlsr.yaml

MMS_ARGS=\
	--asr_tag train_asr_mms_duration \
	--asr_config conf/train_asr_mms.yaml

MMS_DRO_ARGS=\
	--asr_tag train_asr_mms_dro_1_duration \
	--asr_config conf/train_asr_mms_dro_orig.yaml

MMS_DRO2_ARGS=\
	--asr_tag train_asr_mms_dro_01_language \
	--asr_config conf/train_asr_mms_dro_mod.yaml

XLSR_DRO_ARGS=\
	--asr_tag train_asr_xlsr_dro_1_duration \
	--asr_config conf/train_asr_xlsr_dro_orig.yaml

XLSR_DRO2_ARGS=\
	--asr_tag train_asr_xlsr_dro_01_duration \
	--asr_config conf/train_asr_xlsr_dro_mod.yaml

activate-venv:
	source ../../../tools/activate_python.sh 

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


train-xlsr:
	./run_multi.sh \
		$(COMMON_ARGS) \
		$(XLSR_ARGS) \
		--stage 11

train-mms:
	./run_multi.sh \
		$(COMMON_ARGS) \
		$(MMS_ARGS) \
		--stage 11

train-xlsr-dro:
	./run_multi.sh \
		$(COMMON_ARGS) \
		$(XLSR_DRO_ARGS) \
		--stage 11

train-xlsr-dro2:
	./run_multi.sh \
		$(COMMON_ARGS) \
		$(XLSR_DRO2_ARGS) \
		--stage 11

train-mms-dro:
	./run_multi.sh \
		$(COMMON_ARGS) \
		$(MMS_DRO_ARGS) \
		--stage 11

train-mms-dro2:
	./run_multi.sh \
		$(COMMON_ARGS) \
		$(MMS_DRO2_ARGS) \
		--stage 11

# eval commands

# make train-xslr-ctc-aeb
# make train-xslr-ctc-dro-aeb
# make train-xslr-ctc-dro-rm-aeb
# make train-xslr-ctc-ceb
# make train-xslr-ctc-dro-ceb
# make train-xslr-ctc-dro-rm-ceb
# make eval-xslr-ctc-aeb
# make eval-xslr-ctc-dro-aeb
# make eval-xslr-ctc-dro-rm-aeb
# make eval-xslr-ctc-ceb
# make eval-xslr-ctc-dro-ceb
# make eval-xslr-ctc-dro-rm-ceb

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
