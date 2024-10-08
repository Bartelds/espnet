#!/bin/bash

nlprun -n train_asr_mms_aleb_dro_0.001 -g 1 -a /nlp/scr/bartelds/miniconda3/envs/asr-dro -o results/exp_008/train_asr_mms_aleb_dro_0.001.txt --mail-user bartelds@stanford.edu -d a6000 'source ../../../tools/activate_python.sh; make train_asr_mms_aleb_dro_0.001'
nlprun -n train_asr_mms_sceb_dro_0.001_uniforminit -g 1 -a /nlp/scr/bartelds/miniconda3/envs/asr-dro -o results/exp_008/train_asr_mms_sceb_dro_0.001_uniforminit.txt --mail-user bartelds@stanford.edu -d a6000 'source ../../../tools/activate_python.sh; make train_asr_mms_sceb_dro_0.001_uniforminit'
nlprun -n train_asr_xlsr_aleb_dro_0.01 -g 1 -a /nlp/scr/bartelds/miniconda3/envs/asr-dro -o results/exp_008/train_asr_xlsr_aleb_dro_0.01.txt --mail-user bartelds@stanford.edu -d a6000 'source ../../../tools/activate_python.sh; make train_asr_xlsr_aleb_dro_0.01'
nlprun -n train_asr_xlsr_sceb_dro_0.001_uniforminit -g 1 -a /nlp/scr/bartelds/miniconda3/envs/asr-dro -o results/exp_008/train_asr_xlsr_sceb_dro_0.001_uniforminit.txt --mail-user bartelds@stanford.edu -d a6000 'source ../../../tools/activate_python.sh; make train_asr_xlsr_sceb_dro_0.001_uniforminit'
