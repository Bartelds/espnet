.ONESHELL:

include cluster_info.mk
DATA_SUBSET=1h

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
##
# Loss Functions
###
LOSS_CTC_ARGS=\
	--asr_config conf/exp_001/train_asr_xlsr.yaml

LOSS_CTC_DRO_ARGS=\
	--asr_config conf/exp_001/train_asr_xlsr_dro.yaml

LOSS_CTC_DRO_RM_ARGS=\
	--asr_config conf/exp_001/train_asr_xlsr_dro_rm.yaml

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
	--asr_config conf/exp_001/train_asr_xlsr.yaml

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

train-xslr-ctc-dro-rm-aleb:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(LOSS_CTC_DRO_RM_ARGS) $(ALEB_PARAMS)

train-xslr-ctc-sceb:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(LOSS_CTC_ARGS) $(SCEB_PARAMS)

train-xslr-ctc-dro-sceb:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(LOSS_CTC_DRO_ARGS) $(SCEB_PARAMS)

train-xslr-ctc-dro-rm-sceb:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(LOSS_CTC_DRO_RM_ARGS) $(SCEB_PARAMS)


##
# Evaluation for 6 experimental conditions
###
eval-xslr-ctc-aleb:
	echo "TODO"

eval-xslr-ctc-dro-aleb:
	echo "TODO"

eval-xslr-ctc-dro-rm-aleb:
	echo "TODO"

eval-xslr-ctc-sceb:
	echo "TODO"

eval-xslr-ctc-dro-sceb:
	echo "TODO"

eval-xslr-ctc-dro-rm-sceb:
	echo "TODO"




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