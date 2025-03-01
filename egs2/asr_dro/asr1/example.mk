.ONESHELL:

include cluster_info.mk
EXPERIMENT_ID=exp_000
DATA_SUBSET=1h
USER_SCTK_INSTALL_DIR=
SPECIFIC_LANGUAGES=true
SELECTED_LANGUAGES=pol,spa,ces,ron,nan,cmn
DATASETS=M-AILABS,voxforge,commonvoice,fleurs,commonvoice,fleurs

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
	--exp_dir $(EXP_DIR)/asr_train_$(subst eval_,,$@)/decode_asr_asr_model_valid.loss.best/test_1h_lid/score_cer/few_shot/trained/ 

EVAL_CMD=\
	./local/score_macro.sh $(COMMON_EVAL_ARGS) > results/$(EXPERIMENT_ID)/$@.txt

ALEB_PARAMS=\
	--batch_type duration_language 

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

results/$(EXPERIMENT_ID)/:
	mkdir -p results/$(EXPERIMENT_ID)/

activate-venv:
	source ../../../tools/activate_python.sh 

MMS_LOSS_CTC_0.0001_ARGS= --asr_config conf/$(EXPERIMENT_ID)/train_asr_mms_0.0001.yaml

XLSR_LOSS_CTC_0.0001_ARGS= --asr_config conf/$(EXPERIMENT_ID)/train_asr_xlsr_0.0001.yaml

MMS_LOSS_CTC_0.001_ALEB_ARGS= --asr_config conf/$(EXPERIMENT_ID)/train_asr_mms_aleb_dro_0.001.yaml

MMS_LOSS_CTC_0.001_ALEB_smooth_0.1_ARGS= --asr_config conf/$(EXPERIMENT_ID)/train_asr_mms_aleb_dro_0.001_smooth_0.1.yaml

MMS_LOSS_CTC_0.001_ALEB_smooth_0.5_ARGS= --asr_config conf/$(EXPERIMENT_ID)/train_asr_mms_aleb_dro_0.001_smooth_0.5.yaml

MMS_LOSS_CTC_0.001_ALEB_smooth_1.0_ARGS= --asr_config conf/$(EXPERIMENT_ID)/train_asr_mms_aleb_dro_0.001_smooth_1.0.yaml

XLSR_LOSS_CTC_0.001_ALEB_ARGS= --asr_config conf/$(EXPERIMENT_ID)/train_asr_xlsr_aleb_dro_0.001.yaml

XLSR_LOSS_CTC_0.001_ALEB_smooth_0.1_ARGS= --asr_config conf/$(EXPERIMENT_ID)/train_asr_xlsr_aleb_dro_0.001_smooth_0.1.yaml

XLSR_LOSS_CTC_0.001_ALEB_smooth_0.5_ARGS= --asr_config conf/$(EXPERIMENT_ID)/train_asr_xlsr_aleb_dro_0.001_smooth_0.5.yaml

XLSR_LOSS_CTC_0.001_ALEB_smooth_1.0_ARGS= --asr_config conf/$(EXPERIMENT_ID)/train_asr_xlsr_aleb_dro_0.001_smooth_1.0.yaml

MMS_LOSS_CTC_0.0001_ALEB_ARGS= --asr_config conf/$(EXPERIMENT_ID)/train_asr_mms_aleb_dro_0.0001.yaml

MMS_LOSS_CTC_0.0001_ALEB_smooth_0.1_ARGS= --asr_config conf/$(EXPERIMENT_ID)/train_asr_mms_aleb_dro_0.0001_smooth_0.1.yaml

MMS_LOSS_CTC_0.0001_ALEB_smooth_0.5_ARGS= --asr_config conf/$(EXPERIMENT_ID)/train_asr_mms_aleb_dro_0.0001_smooth_0.5.yaml

MMS_LOSS_CTC_0.0001_ALEB_smooth_1.0_ARGS= --asr_config conf/$(EXPERIMENT_ID)/train_asr_mms_aleb_dro_0.0001_smooth_1.0.yaml

XLSR_LOSS_CTC_0.0001_ALEB_ARGS= --asr_config conf/$(EXPERIMENT_ID)/train_asr_xlsr_aleb_dro_0.0001.yaml

XLSR_LOSS_CTC_0.0001_ALEB_smooth_0.1_ARGS= --asr_config conf/$(EXPERIMENT_ID)/train_asr_xlsr_aleb_dro_0.0001_smooth_0.1.yaml

XLSR_LOSS_CTC_0.0001_ALEB_smooth_0.5_ARGS= --asr_config conf/$(EXPERIMENT_ID)/train_asr_xlsr_aleb_dro_0.0001_smooth_0.5.yaml

XLSR_LOSS_CTC_0.0001_ALEB_smooth_1.0_ARGS= --asr_config conf/$(EXPERIMENT_ID)/train_asr_xlsr_aleb_dro_0.0001_smooth_1.0.yaml

train-mms-ctc-aleb-0.0001:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(MMS_LOSS_CTC_0.0001_ARGS) $(ALEB_PARAMS)

train-xlsr-ctc-aleb-0.0001:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(XLSR_LOSS_CTC_0.0001_ARGS) $(ALEB_PARAMS)

train_asr_mms_aleb_dro_0.001_smooth_0.1:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(MMS_LOSS_CTC_0.001_ALEB_smooth_0.1_ARGS) $(ALEB_PARAMS)

train_asr_mms_aleb_dro_0.001_smooth_0.5:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(MMS_LOSS_CTC_0.001_ALEB_smooth_0.5_ARGS) $(ALEB_PARAMS)

train_asr_mms_aleb_dro_0.001_smooth_1.0:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(MMS_LOSS_CTC_0.001_ALEB_smooth_1.0_ARGS) $(ALEB_PARAMS)

train_asr_xlsr_aleb_dro_0.001:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(XLSR_LOSS_CTC_0.001_ALEB_ARGS) $(ALEB_PARAMS)

train_asr_xlsr_aleb_dro_0.001_smooth_0.1:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(XLSR_LOSS_CTC_0.001_ALEB_smooth_0.1_ARGS) $(ALEB_PARAMS)

train_asr_xlsr_aleb_dro_0.001_smooth_0.5:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(XLSR_LOSS_CTC_0.001_ALEB_smooth_0.5_ARGS) $(ALEB_PARAMS)

train_asr_xlsr_aleb_dro_0.001_smooth_1.0:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(XLSR_LOSS_CTC_0.001_ALEB_smooth_1.0_ARGS) $(ALEB_PARAMS)

train_asr_mms_aleb_dro_0.0001_smooth_0.1:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(MMS_LOSS_CTC_0.0001_ALEB_smooth_0.1_ARGS) $(ALEB_PARAMS)

train_asr_mms_aleb_dro_0.0001_smooth_0.5:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(MMS_LOSS_CTC_0.0001_ALEB_smooth_0.5_ARGS) $(ALEB_PARAMS)

train_asr_mms_aleb_dro_0.0001_smooth_1.0:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(MMS_LOSS_CTC_0.0001_ALEB_smooth_1.0_ARGS) $(ALEB_PARAMS)

train_asr_xlsr_aleb_dro_0.0001:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(XLSR_LOSS_CTC_0.0001_ALEB_ARGS) $(ALEB_PARAMS)

train_asr_xlsr_aleb_dro_0.0001_smooth_0.1:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(XLSR_LOSS_CTC_0.0001_ALEB_smooth_0.1_ARGS) $(ALEB_PARAMS)

train_asr_xlsr_aleb_dro_0.0001_smooth_0.5:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(XLSR_LOSS_CTC_0.0001_ALEB_smooth_0.5_ARGS) $(ALEB_PARAMS)

train_asr_xlsr_aleb_dro_0.0001_smooth_1.0:
	./run_multi.sh $(COMMON_TRAIN_ARGS) $(XLSR_LOSS_CTC_0.0001_ALEB_smooth_1.0_ARGS) $(ALEB_PARAMS)

eval-mms-ctc-aleb-0.0001: results/$(EXPERIMENT_ID)/$(EVAL_CMD)

eval-xlsr-ctc-aleb-0.0001: results/$(EXPERIMENT_ID)/$(EVAL_CMD)

eval_asr_mms_aleb_dro_0.001_smooth_0.1: results/$(EXPERIMENT_ID)/

eval_asr_mms_aleb_dro_0.001_smooth_0.5: results/$(EXPERIMENT_ID)/

eval_asr_mms_aleb_dro_0.001_smooth_1.0: results/$(EXPERIMENT_ID)/

eval_asr_xlsr_aleb_dro_0.001_smooth_0.1: results/$(EXPERIMENT_ID)/

eval_asr_xlsr_aleb_dro_0.001_smooth_0.5: results/$(EXPERIMENT_ID)/

eval_asr_xlsr_aleb_dro_0.001_smooth_1.0: results/$(EXPERIMENT_ID)/

eval_asr_mms_aleb_dro_0.0001_smooth_0.1: results/$(EXPERIMENT_ID)/

eval_asr_mms_aleb_dro_0.0001_smooth_0.5: results/$(EXPERIMENT_ID)/

eval_asr_mms_aleb_dro_0.0001_smooth_1.0: results/$(EXPERIMENT_ID)/

eval_asr_xlsr_aleb_dro_0.0001: results/$(EXPERIMENT_ID)/

eval_asr_xlsr_aleb_dro_0.0001_smooth_0.1: results/$(EXPERIMENT_ID)/

eval_asr_xlsr_aleb_dro_0.0001_smooth_0.5: results/$(EXPERIMENT_ID)/

eval_asr_xlsr_aleb_dro_0.0001_smooth_1.0: results/$(EXPERIMENT_ID)/
