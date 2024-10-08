.ONESHELL:

include cluster_info.mk
EXPERIMENT_ID=exp_006_whisper
DATA_SUBSET=1h
USER_SCTK_INSTALL_DIR=
SPECIFIC_LANGUAGES=true
SELECTED_LANGUAGES=afr,eng,spa,sah,nso,tgk,ast,ind,jav,tel,bre,som,isl,urd,kam
DATASETS=nchlt,LAD,mls,commonvoice,nchlt,fleurs,fleurs,commonvoice,googlei18n_asr,fleurs,commonvoice,fleurs,fleurs,fleurs,fleurs

DUMP_DIR=outputs/006/dump
EXP_DIR=outputs/006/exp_subset
ASR_STATS_DIR=outputs/006/exp_subset
USER_SCTK_INSTALL_DIR=


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
	--exp_dir outputs/exp_subset/asr_train-$(subst eval-,,$@)/decode_asr_asr_model_valid.loss.ave/test_1h_lid/score_cer/few_shot/trained/ 

EVAL_CMD=\
	./local/score_macro.sh $(COMMON_EVAL_ARGS) > results/$(EXPERIMENT_ID)/$@.txt

##
# Loss Functions
###
LOSS_ARGS=\
	--asr_config conf/$(EXPERIMENT_ID)/train_whisper.yaml

LOSS_DRO_ARGS=\
	--asr_config conf/$(EXPERIMENT_ID)/train_whisper_dro.yaml

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
	--asr_config conf/$(EXPERIMENT_ID)/train_whisper.yaml

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
# Training for 4 experimental conditions
###
train-whisper-aleb:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(LOSS_ARGS) $(ALEB_PARAMS)

train-whisper-dro-aleb:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(LOSS_DRO_ARGS) $(ALEB_PARAMS)

train-whisper-sceb:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(LOSS_ARGS) $(SCEB_PARAMS)

train-whisper-dro-sceb:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(LOSS_DRO_ARGS) $(SCEB_PARAMS)



##
# Evaluation for 4 experimental conditions
###
results/$(EXPERIMENT_ID)/:
	mkdir -p results/$(EXPERIMENT_ID)/

eval-whisper-aleb: results/$(EXPERIMENT_ID)/
	$(EVAL_CMD)

eval-whisper-dro-aleb: results/$(EXPERIMENT_ID)/
	$(EVAL_CMD)

eval-whisper-sceb: results/$(EXPERIMENT_ID)/
	$(EVAL_CMD)

eval-whisper-dro-sceb: results/$(EXPERIMENT_ID)/
	$(EVAL_CMD)




eval-all: \
	eval-whisper-aleb \
	eval-whisper-dro-aleb \
	eval-whisper-sceb \
	eval-whisper-dro-sceb \
	echo "done"

include cluster-management.mk
