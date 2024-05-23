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
DUMP_DIR=/nlp/scr/ananjan/asrdro/dump
EXP_DIR=/nlp/scr/ananjan/asrdro/exp_subset
ASR_STATS_DIR=/nlp/scr/ananjan/asrdro/exp_subset

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
	--asr_config conf/train_asr_xlsr.yaml

MMS_ARGS=\
	--asr_config conf/train_asr_mms.yaml

XLSR_DRO_ARGS=\
	--asr_config conf/train_asr_xlsr_dro.yaml

activate-venv:
	../../../tools/activate_python.sh 

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

train-dro:
	./run_multi.sh \
		$(COMMON_ARGS) \
		$(XLSR_DRO_ARGS) \
		--stage 11


submit-target-to-cluster:
	echo "#!/bin/bash" > slurm_job.sh
	echo "cd `pwd`" >> slurm_job.sh
	echo "make $(TARGET)" >> slurm_job.sh
	sbatch \
			--time=120:00:00 \
			--cpus-per-task=64 \
			--gres=gpu:4 \
			--mem 128G \
			--account $(SLURM_CLUSTER_ACCOUNT_NAME) \
			--partition $(SLURM_CLUSTER_PARTITION_NAME) \
			--job-name $(TARGET) \
			--exclude=$(SLURM_CLUSTER_EXCLUDE_NODES) \
			slurm_job.sh 

show-jobs:
	squeue --format="%.18i %.12P %70j %.8T %.10M %.9l %.6D %R" -u moussa
