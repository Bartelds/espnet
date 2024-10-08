STEP_SIZE = [0.1, 0.01, 0.001]
BATCHING = {"aleb":"ALEB", "sceb":"SCEB"}
MODELS = {'mms':'MMS', 'xlsr':'XLSR'}
USER = 'bartelds@stanford.edu'
ENV = '/nlp/scr/bartelds/miniconda3/envs/asr-dro'

bash_file = "!/bin/bash\n\n"

idx = 0

for step_size in STEP_SIZE:
    for model in MODELS.keys():
        bash_file += f"nlprun -n train_asr_{model}_sceb_dro_{step_size}_groupinit -g 1 -a {ENV} -o results/exp_006/train_asr_{model}_sceb_dro_{step_size}_groupinit.txt --mail-user {USER} -d a6000 'source ../../../tools/activate_python.sh; make train_asr_{model}_sceb_dro_{step_size}_groupinit'\n"
        bash_file += f"nlprun -n train_asr_{model}_sceb_dro_{step_size}_uniforminit -g 1 -a {ENV} -o results/exp_006/train_asr_{model}_sceb_dro_{step_size}_uniforminit.txt --mail-user {USER} -d a6000 'source ../../../tools/activate_python.sh; make train_asr_{model}_sceb_dro_{step_size}_uniforminit'\n"
        bash_file += f"nlprun -n train_asr_{model}_aleb_dro_{step_size} -g 1 -a {ENV} -o results/exp_006/train_asr_{model}_aleb_dro_{step_size}.txt --mail-user {USER} -d a6000 'source ../../../tools/activate_python.sh; make train_asr_{model}_aleb_dro_{step_size}'\n"

with open('commands_auto_dro.sh', 'w') as f:
    f.write(bash_file)