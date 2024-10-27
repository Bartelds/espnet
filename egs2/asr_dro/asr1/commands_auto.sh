!/bin/bash

nlprun -n train-mms-ctc-aleb-0.0001 -g 1 -a asrdro -o results/exp041/train-mms-ctc-aleb-0.0001.txt --mail-user ananjan -d a6000 'source ../../../tools/activate_python.sh; make train_mms_ctc_aleb_0.0001'
