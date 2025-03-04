LR = [1e-4]
BATCHING = {"aleb":"ALEB"}
# MODELS = {'xlsr':'XLSR'}
MODELS = {'xlsr':'XLSR', 'mms':'MMS'}
SEED = [1,2]
EXP='exp051'

file = open(f'{EXP}_m.mk', 'r').read()
file += '\n\n'

for lr in LR:
    for model in MODELS.keys():
        for seed in SEED:
            file += f"{MODELS[model]}_LOSS_CTC_{float(lr)}_ARGS_{seed}= --asr_config conf/$(EXPERIMENT_ID)/train_asr_{model}_{lr}_{seed}.yaml\n\n"

for lr in LR:
    for batching in BATCHING.keys():
        for model in MODELS.keys():
            for seed in SEED:
                file += f"train_{model}_ctc_{batching}_{float(lr)}_{seed}:\n\t./run_multi.sh $(COMMON_TRAIN_ARGS) $({MODELS[model]}_LOSS_CTC_{float(lr)}_ARGS_{seed}) $({BATCHING[batching]}_PARAMS)\n\n"

for lr in LR:
    for batching in BATCHING.keys():
        for model in MODELS.keys():
            for seed in SEED:
                file += f"eval_{model}_ctc_{batching}_{float(lr)}_{seed}: results/$(EXPERIMENT_ID)/\n\t$(EVAL_CMD)\n\n"

file += "eval-all: /\n"
for lr in LR:
    for batching in BATCHING.keys():
        for model in MODELS.keys():
            for seed in SEED:
                file += f"\tmake eval_{model}_ctc_{batching}_{float(lr)}_{seed} /\n"
file += "\techo 'All done'\n\n"

with open(f'{EXP}.mk', 'w') as f:
    f.write(file)

bash_file = "!/bin/bash\n\n"

idx = 0

for lr in LR:
    for batching in BATCHING.keys():
        for model in MODELS.keys():
            for seed in SEED:
                idx += 1
                if idx <= 0:
                    bash_file += f"nlprun -n train-{model}-ctc-{batching}-{lr}-{seed} -g 1 -a asrdro -o results/{EXP}/train-{model}-ctc-{batching}-{lr}-{seed}.txt --mail-user ananjan -d a6000 -p high 'source ../../../tools/activate_python.sh; make train_{model}_ctc_{batching}_{lr}_{seed}' -p high\n"
                else:
                    bash_file += f"nlprun -n train-{model}-ctc-{batching}-{lr}-{seed} -g 1 -a asrdro -o results/{EXP}/train-{model}-ctc-{batching}-{lr}-{seed}.txt --mail-user ananjan -d a6000 'source ../../../tools/activate_python.sh; make train_{model}_ctc_{batching}_{lr}_{seed}' -p high\n"

with open('commands_auto.sh', 'w') as f:
    f.write(bash_file)
