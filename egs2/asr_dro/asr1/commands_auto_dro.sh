!/bin/bash

nlprun -n train_asr_mms_aleb_dro_0.01 -g 1 -a asrdro -o results/exp043/train_asr_mms_aleb_dro_0.01.txt --mail-user ananjan -d a6000 'source ../../../tools/activate_python.sh; make train_asr_mms_aleb_dro_0.01'
nlprun -n train_asr_mms_aleb_dro_0.001 -g 1 -a asrdro -o results/exp043/train_asr_mms_aleb_dro_0.001.txt --mail-user ananjan -d a6000 'source ../../../tools/activate_python.sh; make train_asr_mms_aleb_dro_0.001'
