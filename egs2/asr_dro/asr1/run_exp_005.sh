LR = [5e-4,1e-4,5e-5,1e-5]
BATCHING = {"aleb":"ALEB", "sceb":"SCEB"}
MODELS = {'mms':'MMS', 'xlsr':'XLSR'}
USER = 'bartelds@stanford.edu'
ENV = '/nlp/scr/bartelds/miniconda3/envs/asr-dro'

bash_file = "!/bin/bash\n\n"
idx = 0

for lr in LR:
    for batching in BATCHING.keys():
        for model in MODELS.keys():
            idx += 1
            if idx <= 5:
                bash_file += f"nlprun -n train-{model}-ctc-{batching}-{lr} -g 1 -a {ENV} -o results/exp_005/train-{model}-ctc-{batching}-{lr}.txt --mail-user {USER} -d a6000 -p high 'source ../../../tools/activate_python.sh; make train-{model}-ctc-{batching}-{lr}'\n"
            else:
                bash_file += f"nlprun -n train-{model}-ctc-{batching}-{lr} -g 1 -a {ENV} -o results/exp_005/train-{model}-ctc-{batching}-{lr}.txt --mail-user {USER} -d a6000 'source ../../../tools/activate_python.sh; make train-{model}-ctc-{batching}-{lr}'\n"

with open('commands_auto.sh', 'w') as f:
    f.write(bash_file)
