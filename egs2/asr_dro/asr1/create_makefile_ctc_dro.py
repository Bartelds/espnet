STEP_SIZE = [0.0001, 0.001]
SMOOTHING = [0.1, 0.5, 1.0]
MODELS = {'mms':'MMS', 'xlsr':'XLSR'}
EXP='exp001'

file = open(f'{EXP}_m.mk', 'r').read()
file += '\n\n'

for step_size in STEP_SIZE:
    for smoothing in SMOOTHING:
        for model in MODELS.keys():
            file += f"{MODELS[model]}_LOSS_CTC_{float(step_size)}_LA_{float(smoothing)}_ALEB_ARGS= --asr_config conf/$(EXPERIMENT_ID)/{model}_example_ctc_dro_{step_size}_la_{smoothing}.yaml\n\n"

for step_size in STEP_SIZE:
    for smoothing in SMOOTHING:
        for model in MODELS.keys():
            file += f"train_asr_{model}_aleb_dro_{step_size}_la_{smoothing}:\n\t./run_multi.sh $(COMMON_TRAIN_ARGS) $({MODELS[model]}_LOSS_CTC_{float(step_size)}_LA_{float(smoothing)}_ALEB_ARGS) $(ALEB_PARAMS)\n\n"

for step_size in STEP_SIZE:
    for smoothing in SMOOTHING:
        for model in MODELS.keys():
            file += f"eval_asr_{model}_aleb_dro_{step_size}_la_{smoothing}: results/$(EXPERIMENT_ID)/\n\t$(EVAL_CMD)\n\n"

file += "eval-all: /\n"
for step_size in STEP_SIZE:
    for smoothing in SMOOTHING:
        for model in MODELS.keys():
            file += f"\tmake eval_asr_{model}_aleb_dro_{step_size}_la_{smoothing} /\n"
file += "\techo 'All done'\n\n"

with open(f'{EXP}_auto_ctc_dro.mk', 'w') as f:
    f.write(file)