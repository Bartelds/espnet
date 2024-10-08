LR = [1e-4]
BATCHING = {"aleb":"ALEB", "sceb":"SCEB"}
MODELS = {'mms':'MMS', 'xlsr':'XLSR'}
# MODELS = {'mms':'MMS'}

file = open('exp005_m.mk', 'r').read()
file += '\n\n'

for lr in LR:
    for model in MODELS.keys():
        file += f"{MODELS[model]}_LOSS_CTC_{float(lr)}_ARGS= --asr_config conf/$(EXPERIMENT_ID)/train_asr_{model}_{lr}.yaml\n\n"

for lr in LR:
    for batching in BATCHING.keys():
        for model in MODELS.keys():
            file += f"train-{model}-ctc-{batching}-{float(lr)}:\n\t./run_multi.sh $(COMMON_TRAIN_ARGS) $({MODELS[model]}_LOSS_CTC_{float(lr)}_ARGS) $({BATCHING[batching]}_PARAMS)\n\n"

for lr in LR:
    for batching in BATCHING.keys():
        for model in MODELS.keys():
            file += f"eval-{model}-ctc-{batching}-{float(lr)}: results/$(EXPERIMENT_ID)/\n\t$(EVAL_CMD)\n\n"

file += "eval-all: /\n"
for lr in LR:
    for batching in BATCHING.keys():
        for model in MODELS.keys():
            file += f"\tmake eval-{model}-ctc-{batching}-{float(lr)} /\n"
file += "\techo 'All done'\n\n"

with open('exp005_auto.mk', 'w') as f:
    f.write(file)

bash_file = "!/bin/bash\n\n"

idx = 0

for lr in LR:
    for batching in BATCHING.keys():
        for model in MODELS.keys():
            idx += 1
            if idx <= 0:
                bash_file += f"nlprun -n train-{model}-ctc-{batching}-{lr} -g 1 -a asrdro -o results/exp-005/train-{model}-ctc-{batching}-{lr}.txt --mail-user ananjan -d a6000 -p high 'source ../../../tools/activate_python.sh; make train-{model}-ctc-{batching}-{lr}'\n"
            else:
                bash_file += f"nlprun -n train-{model}-ctc-{batching}-{lr} -g 1 -a asrdro -o results/exp-005/train-{model}-ctc-{batching}-{lr}.txt --mail-user ananjan -d a6000 'source ../../../tools/activate_python.sh; make train-{model}-ctc-{batching}-{lr}'\n"

with open('commands_auto.sh', 'w') as f:
    f.write(bash_file)
