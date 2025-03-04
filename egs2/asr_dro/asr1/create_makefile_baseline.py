LR = [1e-4]
MODELS = {'xlsr':'XLSR', 'mms':'MMS'}
EXP='exp001'

file = open(f'{EXP}_m.mk', 'r').read()
file += '\n\n'

for lr in LR:
    for model in MODELS.keys():
        file += f"{MODELS[model]}_LOSS_CTC_{float(lr)}_ARGS= --asr_config conf/$(EXPERIMENT_ID)/{model}_example_baseline_{lr}.yaml\n\n"

for lr in LR:
    for model in MODELS.keys():
            file += f"train_{model}_ctc_aleb_{float(lr)}:\n\t./run_multi.sh $(COMMON_TRAIN_ARGS) $({MODELS[model]}_LOSS_CTC_{float(lr)}_ARGS) $(ALEB_PARAMS)\n\n"

for lr in LR:
    for model in MODELS.keys():
        file += f"eval_{model}_ctc_aleb_{float(lr)}: results/$(EXPERIMENT_ID)/\n\t$(EVAL_CMD)\n\n"

file += "eval-all: /\n"
for lr in LR:
    for model in MODELS.keys():
        file += f"\tmake eval_{model}_ctc_aleb_{float(lr)} /\n"
file += "\techo 'All done'\n\n"

with open(f'{EXP}_auto_baseline.mk', 'w') as f:
    f.write(file)
