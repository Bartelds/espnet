.ONESHELL:

include cluster_info.mk
DATA_SUBSET=1h
DUMP_DIR=outputs/dump
EXP_DIR=outputs/exp_subset
ASR_STATS_DIR=outputs/exp_subset
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
	./local/score_macro.sh $(COMMON_EVAL_ARGS) > results/exp-001/$@.txt

##
# Loss Functions
###
LOSS_CTC_ARGS=\
	--asr_config conf/exp_001/train_asr_xlsr.yaml

LOSS_CTC_DRO_ARGS=\
	--asr_config conf/exp_001/train_asr_xlsr_dro.yaml

LOSS_CTC_DRO_RM_ARGS=\
	--asr_config conf/exp_001/train_asr_xlsr_dro_rm.yaml

LOSS_CTC_DRO_HPTUNE_001_ARGS=\
	--asr_config conf/exp_001/train_asr_xlsr_dro_hptune_001.yaml

LOSS_CTC_DRO_RM_HPTUNE_001_ARGS=\
	--asr_config conf/exp_001/train_asr_xlsr_dro_rm_hptune_001.yaml

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
# Training for 4 hparam tunins conditions
###
train-xslr-ctc-dro-aleb-hptune-001:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(LOSS_CTC_DRO_HPTUNE_001_ARGS) $(ALEB_PARAMS)

train-xslr-ctc-dro-rm-aleb-hptune-001:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(LOSS_CTC_DRO_RM_HPTUNE_001_ARGS) $(ALEB_PARAMS)

train-xslr-ctc-dro-sceb-hptune-001:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(LOSS_CTC_DRO_HPTUNE_001_ARGS) $(SCEB_PARAMS)

train-xslr-ctc-dro-rm-sceb-hptune-001:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(LOSS_CTC_DRO_RM_HPTUNE_001_ARGS) $(SCEB_PARAMS)


##
# Evaluation for 10 experimental conditions
###
results/exp-001/:
	mkdir -p results/exp-001/

eval-xslr-ctc-aleb: results/exp-001/
	$(EVAL_CMD)

eval-xslr-ctc-dro-aleb: results/exp-001/
	$(EVAL_CMD)

eval-xslr-ctc-dro-rm-aleb: results/exp-001/
	$(EVAL_CMD)

eval-xslr-ctc-sceb: results/exp-001/
	$(EVAL_CMD)

eval-xslr-ctc-dro-sceb: results/exp-001/
	$(EVAL_CMD)

eval-xslr-ctc-dro-rm-sceb: results/exp-001/
	$(EVAL_CMD)



eval-xslr-ctc-dro-aleb-hptune-001: results/exp-001/
	$(EVAL_CMD)
	
eval-xslr-ctc-dro-rm-aleb-hptune-001: results/exp-001/
	$(EVAL_CMD)

eval-xslr-ctc-dro-sceb-hptune-001: results/exp-001/
	$(EVAL_CMD)

eval-xslr-ctc-dro-rm-sceb-hptune-001: results/exp-001/
	$(EVAL_CMD)


eval-all: \
	eval-xslr-ctc-aleb \
	eval-xslr-ctc-dro-aleb \
	eval-xslr-ctc-dro-rm-aleb \
	eval-xslr-ctc-sceb \
	eval-xslr-ctc-dro-sceb \
	eval-xslr-ctc-dro-rm-sceb
	echo "done"

include cluster-management.mk