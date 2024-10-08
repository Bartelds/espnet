#!/bin/bash

nlprun -n train-mms-ctc-aleb-0.0001 -g 1 -a /nlp/scr/bartelds/miniconda3/envs/asr-dro -o results/exp_037/train-mms-ctc-aleb-0.0001.txt --mail-user bartelds@stanford.edu -d a6000 -x jagupard36,jagupard35 -r 64G -c 4 'source ../../../tools/activate_python.sh; make train-mms-ctc-aleb-0.0001'
nlprun -n train-xlsr-ctc-aleb-0.0001 -g 1 -a /nlp/scr/bartelds/miniconda3/envs/asr-dro -o results/exp_037/train-xlsr-ctc-aleb-0.0001.txt --mail-user bartelds@stanford.edu -d a6000 -x jagupard36,jagupard35 -r 64G -c 4 'source ../../../tools/activate_python.sh; make train-xlsr-ctc-aleb-0.0001'
nlprun -n train-mms-ctc-sceb-0.0001 -g 1 -a /nlp/scr/bartelds/miniconda3/envs/asr-dro -o results/exp_037/train-mms-ctc-sceb-0.0001.txt --mail-user bartelds@stanford.edu -d a6000 -x jagupard36,jagupard35 -r 64G -c 4 'source ../../../tools/activate_python.sh; make train-mms-ctc-sceb-0.0001'
nlprun -n train-xlsr-ctc-sceb-0.0001 -g 1 -a /nlp/scr/bartelds/miniconda3/envs/asr-dro -o results/exp_037/train-xlsr-ctc-sceb-0.0001.txt --mail-user bartelds@stanford.edu -d a6000 -x jagupard36,jagupard35 -r 64G -c 4 'source ../../../tools/activate_python.sh; make train-xlsr-ctc-sceb-0.0001'
