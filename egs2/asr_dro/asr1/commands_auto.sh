!/bin/bash

nlprun -n train-mms-ctc-aleb-0.0001 -g 1 -a asrdro -o results/exp-005/train-mms-ctc-aleb-0.0001.txt --mail-user ananjan -d a6000 'source ../../../tools/activate_python.sh; make train-mms-ctc-aleb-0.0001'
nlprun -n train-xlsr-ctc-aleb-0.0001 -g 1 -a asrdro -o results/exp-005/train-xlsr-ctc-aleb-0.0001.txt --mail-user ananjan -d a6000 'source ../../../tools/activate_python.sh; make train-xlsr-ctc-aleb-0.0001'
nlprun -n train-mms-ctc-sceb-0.0001 -g 1 -a asrdro -o results/exp-005/train-mms-ctc-sceb-0.0001.txt --mail-user ananjan -d a6000 'source ../../../tools/activate_python.sh; make train-mms-ctc-sceb-0.0001'
nlprun -n train-xlsr-ctc-sceb-0.0001 -g 1 -a asrdro -o results/exp-005/train-xlsr-ctc-sceb-0.0001.txt --mail-user ananjan -d a6000 'source ../../../tools/activate_python.sh; make train-xlsr-ctc-sceb-0.0001'
