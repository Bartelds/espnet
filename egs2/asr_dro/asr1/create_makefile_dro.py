# STEP_SIZE = [0.1, 0.01, 0.001]
STEP_SIZE = [0.01, 0.1]
# STEP_SIZE = [0]
BATCHING = {"aleb":"ALEB", "sceb":"SCEB"}
MODELS = {'mms':'MMS', 'xlsr':'XLSR'}
# MODELS = {'mms':'MMS'}

file = open('exp005_m.mk', 'r').read()
file += '\n\n'

for step_size in STEP_SIZE:
    for model in MODELS.keys():
        file += f"{MODELS[model]}_LOSS_CTC_{float(step_size)}_SCEB_GROUPINIT_ARGS= --asr_config conf/$(EXPERIMENT_ID)/train_asr_{model}_sceb_dro_{step_size}_groupinit.yaml\n\n"
        # file += f"{MODELS[model]}_LOSS_CTC_{float(step_size)}_SCEB_UNIFORMINIT_ARGS= --asr_config conf/$(EXPERIMENT_ID)/train_asr_{model}_sceb_dro_{step_size}_uniforminit.yaml\n\n"
        file += f"{MODELS[model]}_LOSS_CTC_{float(step_size)}_ALEB_ARGS= --asr_config conf/$(EXPERIMENT_ID)/train_asr_{model}_aleb_dro_{step_size}.yaml\n\n"
        # file += f"{MODELS[model]}_LOSS_CTC_{float(step_size)}_SCEB_GROUPINIT_RM_ARGS= --asr_config conf/$(EXPERIMENT_ID)/train_asr_{model}_sceb_dro_{step_size}_groupinit_rm.yaml\n\n"
        # file += f"{MODELS[model]}_LOSS_CTC_{float(step_size)}_SCEB_UNIFORMINIT_RM_ARGS= --asr_config conf/$(EXPERIMENT_ID)/train_asr_{model}_sceb_dro_{step_size}_uniforminit_rm.yaml\n\n"
        # file += f"{MODELS[model]}_LOSS_CTC_{float(step_size)}_ALEB_RM_ARGS= --asr_config conf/$(EXPERIMENT_ID)/train_asr_{model}_aleb_dro_{step_size}_rm.yaml\n\n"

for step_size in STEP_SIZE:
    for model in MODELS.keys():
        file += f"train_asr_{model}_sceb_dro_{step_size}_groupinit:\n\t./run_multi.sh $(COMMON_TRAIN_ARGS) $({MODELS[model]}_LOSS_CTC_{float(step_size)}_SCEB_GROUPINIT_ARGS) $(SCEB_PARAMS)\n\n"
        # file += f"train_asr_{model}_sceb_dro_{step_size}_uniforminit:\n\t./run_multi.sh $(COMMON_TRAIN_ARGS) $({MODELS[model]}_LOSS_CTC_{float(step_size)}_SCEB_UNIFORMINIT_ARGS) $(SCEB_PARAMS)\n\n"
        file += f"train_asr_{model}_aleb_dro_{step_size}:\n\t./run_multi.sh $(COMMON_TRAIN_ARGS) $({MODELS[model]}_LOSS_CTC_{float(step_size)}_ALEB_ARGS) $(ALEB_PARAMS)\n\n"
        # file += f"train_asr_{model}_sceb_dro_{step_size}_groupinit_rm:\n\t./run_multi.sh $(COMMON_TRAIN_ARGS) $({MODELS[model]}_LOSS_CTC_{float(step_size)}_SCEB_GROUPINIT_RM_ARGS) $(SCEB_PARAMS)\n\n"
        # file += f"train_asr_{model}_sceb_dro_{step_size}_uniforminit_rm:\n\t./run_multi.sh $(COMMON_TRAIN_ARGS) $({MODELS[model]}_LOSS_CTC_{float(step_size)}_SCEB_UNIFORMINIT_RM_ARGS) $(SCEB_PARAMS)\n\n"
        # file += f"train_asr_{model}_aleb_dro_{step_size}_rm:\n\t./run_multi.sh $(COMMON_TRAIN_ARGS) $({MODELS[model]}_LOSS_CTC_{float(step_size)}_ALEB_RM_ARGS) $(ALEB_PARAMS)\n\n"

for step_size in STEP_SIZE:
    for model in MODELS.keys():
        file += f"eval_asr_{model}_sceb_dro_{step_size}_groupinit: results/$(EXPERIMENT_ID)/\n\t$(EVAL_CMD)\n\n"
        # file += f"eval_asr_{model}_sceb_dro_{step_size}_uniforminit: results/$(EXPERIMENT_ID)/\n\t$(EVAL_CMD)\n\n"
        file += f"eval_asr_{model}_aleb_dro_{step_size}: results/$(EXPERIMENT_ID)/\n\t$(EVAL_CMD)\n\n"
        # file += f"eval_asr_{model}_sceb_dro_{step_size}_groupinit_rm: results/$(EXPERIMENT_ID)/\n\t$(EVAL_CMD)\n\n"
        # file += f"eval_asr_{model}_sceb_dro_{step_size}_uniforminit_rm: results/$(EXPERIMENT_ID)/\n\t$(EVAL_CMD)\n\n"
        # file += f"eval_asr_{model}_aleb_dro_{step_size}_rm: results/$(EXPERIMENT_ID)/\n\t$(EVAL_CMD)\n\n"

file += "eval-all: /\n"
for step_size in STEP_SIZE:
    for model in MODELS.keys():
        file += f"\tmake eval_asr_{model}_sceb_dro_{step_size}_groupinit /\n"
        # file += f"\tmake eval_asr_{model}_sceb_dro_{step_size}_uniforminit /\n"
        file += f"\tmake eval_asr_{model}_aleb_dro_{step_size} /\n"
        # file += f"\tmake eval_asr_{model}_sceb_dro_{step_size}_groupinit_rm /\n"
        # file += f"\tmake eval_asr_{model}_sceb_dro_{step_size}_uniforminit_rm /\n"
        # file += f"\tmake eval_asr_{model}_aleb_dro_{step_size}_rm /\n"
file += "\techo 'All done'\n\n"

with open('exp005_auto_dro.mk', 'w') as f:
    f.write(file)

bash_file = "!/bin/bash\n\n"

idx = 0

for step_size in STEP_SIZE:
    for model in MODELS.keys():
        bash_file += f"nlprun -n train_asr_{model}_sceb_dro_{step_size}_groupinit -g 1 -a asrdro -o results/exp-005/train_asr_{model}_sceb_dro_{step_size}_groupinit.txt --mail-user ananjan -d a6000 'source ../../../tools/activate_python.sh; make train_asr_{model}_sceb_dro_{step_size}_groupinit'\n"
        # bash_file += f"nlprun -n train_asr_{model}_sceb_dro_{step_size}_uniforminit -g 1 -a asrdro -o results/exp-006/train_asr_{model}_sceb_dro_{step_size}_uniforminit.txt --mail-user ananjan -d a6000 'source ../../../tools/activate_python.sh; make train_asr_{model}_sceb_dro_{step_size}_uniforminit'\n"
        bash_file += f"nlprun -n train_asr_{model}_aleb_dro_{step_size} -g 1 -a asrdro -o results/exp-005/train_asr_{model}_aleb_dro_{step_size}.txt --mail-user ananjan -d a6000 'source ../../../tools/activate_python.sh; make train_asr_{model}_aleb_dro_{step_size}'\n"
        # bash_file += f"nlprun -n train_asr_{model}_sceb_dro_{step_size}_groupinit_rm -g 1 -a asrdro -o results/exp-004/train_asr_{model}_sceb_dro_{step_size}_groupinit_rm.txt --mail-user ananjan -d a6000 'source ../../../tools/activate_python.sh; make train_asr_{model}_sceb_dro_{step_size}_groupinit_rm'\n"
        # bash_file += f"nlprun -n train_asr_{model}_sceb_dro_{step_size}_uniforminit_rm -g 1 -a asrdro -o results/exp-004/train_asr_{model}_sceb_dro_{step_size}_uniforminit_rm.txt --mail-user ananjan -d a6000 'source ../../../tools/activate_python.sh; make train_asr_{model}_sceb_dro_{step_size}_uniforminit_rm'\n"
        # bash_file += f"nlprun -n train_asr_{model}_aleb_dro_{step_size}_rm -g 1 -a asrdro -o results/exp-004/train_asr_{model}_aleb_dro_{step_size}_rm.txt --mail-user ananjan -d a6000 'source ../../../tools/activate_python.sh; make train_asr_{model}_aleb_dro_{step_size}_rm'\n"

with open('commands_auto_dro.sh', 'w') as f:
    f.write(bash_file)
