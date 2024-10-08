!/bin/bash

nlprun -n train_asr_mms_sceb_dro_0.01_groupinit -g 1 -a asrdro -o results/exp-005/train_asr_mms_sceb_dro_0.01_groupinit.txt --mail-user ananjan -d a6000 'source ../../../tools/activate_python.sh; make train_asr_mms_sceb_dro_0.01_groupinit'
nlprun -n train_asr_mms_aleb_dro_0.01 -g 1 -a asrdro -o results/exp-005/train_asr_mms_aleb_dro_0.01.txt --mail-user ananjan -d a6000 'source ../../../tools/activate_python.sh; make train_asr_mms_aleb_dro_0.01'
nlprun -n train_asr_xlsr_sceb_dro_0.01_groupinit -g 1 -a asrdro -o results/exp-005/train_asr_xlsr_sceb_dro_0.01_groupinit.txt --mail-user ananjan -d a6000 'source ../../../tools/activate_python.sh; make train_asr_xlsr_sceb_dro_0.01_groupinit'
nlprun -n train_asr_xlsr_aleb_dro_0.01 -g 1 -a asrdro -o results/exp-005/train_asr_xlsr_aleb_dro_0.01.txt --mail-user ananjan -d a6000 'source ../../../tools/activate_python.sh; make train_asr_xlsr_aleb_dro_0.01'
nlprun -n train_asr_mms_sceb_dro_0.1_groupinit -g 1 -a asrdro -o results/exp-005/train_asr_mms_sceb_dro_0.1_groupinit.txt --mail-user ananjan -d a6000 'source ../../../tools/activate_python.sh; make train_asr_mms_sceb_dro_0.1_groupinit'
nlprun -n train_asr_mms_aleb_dro_0.1 -g 1 -a asrdro -o results/exp-005/train_asr_mms_aleb_dro_0.1.txt --mail-user ananjan -d a6000 'source ../../../tools/activate_python.sh; make train_asr_mms_aleb_dro_0.1'
nlprun -n train_asr_xlsr_sceb_dro_0.1_groupinit -g 1 -a asrdro -o results/exp-005/train_asr_xlsr_sceb_dro_0.1_groupinit.txt --mail-user ananjan -d a6000 'source ../../../tools/activate_python.sh; make train_asr_xlsr_sceb_dro_0.1_groupinit'
nlprun -n train_asr_xlsr_aleb_dro_0.1 -g 1 -a asrdro -o results/exp-005/train_asr_xlsr_aleb_dro_0.1.txt --mail-user ananjan -d a6000 'source ../../../tools/activate_python.sh; make train_asr_xlsr_aleb_dro_0.1'
