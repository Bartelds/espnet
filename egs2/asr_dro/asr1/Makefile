.ONESHELL:

include cluster_info.mk
# SLURM_CLUSTER_ACCOUNT_NAME=cluster-account-name
# SLURM_CLUSTER_PARTITION_NAME=cluster-partition-name
# SLURM_CLUSTER_EXCLUDE_NODES=node1,node2,node3


train-asr-dro-baseline:
	/nlp/scr/moussa/git/asr-dro/espnet/tools/activate_python.sh
	cd /nlp/scr/moussa/git/asr-dro/espnet/egs2/asr_dro/asr1
	./run_multi.sh --asr_config conf/train_asr_dro.yaml --duration 1h --lid true --stage 5


COMMON_ARGS=\
	--duration 10min \
	--lid true \
	--only_lid false \
	--dumpdir outputs/dump \
	--expdir outputs/exp

PREPROCESS_ARGS=\
	--asr_config conf/train_asr.yaml

XLSR_ARGS=\
	--asr_config conf/train_asr_xlsr.yaml

XLSR_DRO_ARGS=\
	--asr_config conf/train_asr_xlsr_dro.yaml

activate-venv:
	../../../tools/activate_python.sh 

	
preprocess:
	./run_multi.sh \
		$(COMMON_ARGS) \
		$(PREPROCESS_ARGS) \
		--stop_stage 10

train-baseline:
	./run_multi.sh \
		$(COMMON_ARGS) \
		$(XLSR_ARGS) \
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
