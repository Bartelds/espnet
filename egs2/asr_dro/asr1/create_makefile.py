LR = [1e-4]
BATCHING = {"aleb":"ALEB"}
MODELS = {'mms':'MMS'}
# MODELS = {'mms':'MMS'}
EXP='exp046'

file = open(f'{EXP}_m.mk', 'r').read()
# file += '\n\n'

for lr in LR:
    for model in MODELS.keys():
        file += f"{MODELS[model]}_LOSS_CTC_{float(lr)}_ARGS= --asr_config conf/$(EXPERIMENT_ID)/train_asr_{model}_{lr}.yaml\n\n"

for lr in LR:
    for batching in BATCHING.keys():
        for model in MODELS.keys():
            file += f"train_{model}_ctc_{batching}_{float(lr)}:\n\t./run_multi.sh $(COMMON_TRAIN_ARGS) $({MODELS[model]}_LOSS_CTC_{float(lr)}_ARGS) $({BATCHING[batching]}_PARAMS)\n\n"

for lr in LR:
    for batching in BATCHING.keys():
        for model in MODELS.keys():
            file += f"eval_{model}_ctc_{batching}_{float(lr)}: results/$(EXPERIMENT_ID)/\n\t$(EVAL_CMD)\n\n"

file += "eval-all: /\n"
for lr in LR:
    for batching in BATCHING.keys():
        for model in MODELS.keys():
            file += f"\tmake eval_{model}_ctc_{batching}_{float(lr)} /\n"
file += "\techo 'All done'\n\n"

with open(f'{EXP}.mk', 'w') as f:
    f.write(file)

bash_file = "!/bin/bash\n\n"

idx = 0

for lr in LR:
    for batching in BATCHING.keys():
        for model in MODELS.keys():
            idx += 1
            if idx <= 0:
                bash_file += f"nlprun -n train-{model}-ctc-{batching}-{lr} -g 1 -a asrdro -o results/{EXP}/train-{model}-ctc-{batching}-{lr}.txt --mail-user ananjan -d a6000 -p high 'source ../../../tools/activate_python.sh; make train_{model}_ctc_{batching}_{lr}'\n"
            else:
                bash_file += f"nlprun -n train-{model}-ctc-{batching}-{lr} -g 1 -a asrdro -o results/{EXP}/train-{model}-ctc-{batching}-{lr}.txt --mail-user ananjan -d a6000 'source ../../../tools/activate_python.sh; make train_{model}_ctc_{batching}_{lr}'\n"

with open('commands_auto.sh', 'w') as f:
    f.write(bash_file)
