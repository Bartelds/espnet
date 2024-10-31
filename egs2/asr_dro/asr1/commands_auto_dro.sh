!/bin/bash

nlprun -n train_asr_mms_aleb_dro_0.0001 -g 1 -a asrdro -o results/exp049/train_asr_mms_aleb_dro_0.0001.txt --mail-user ananjan -m jagupard32 'source ../../../tools/activate_python.sh; make train_asr_mms_aleb_dro_0.0001'
nlprun -n train_asr_mms_aleb_dro_0.001 -g 1 -a asrdro -o results/exp049/train_asr_mms_aleb_dro_0.001.txt --mail-user ananjan -m jagupard32 'source ../../../tools/activate_python.sh; make train_asr_mms_aleb_dro_0.001'
