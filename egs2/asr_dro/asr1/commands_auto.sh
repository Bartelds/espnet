!/bin/bash

nlprun -n train-xlsr-ctc-aleb-0.0001-1 -g 1 -a asrdro -o results/exp051/train-xlsr-ctc-aleb-0.0001-1.txt --mail-user ananjan -d a6000 'source ../../../tools/activate_python.sh; make train_xlsr_ctc_aleb_0.0001_1' -p high
nlprun -n train-xlsr-ctc-aleb-0.0001-2 -g 1 -a asrdro -o results/exp051/train-xlsr-ctc-aleb-0.0001-2.txt --mail-user ananjan -d a6000 'source ../../../tools/activate_python.sh; make train_xlsr_ctc_aleb_0.0001_2' -p high
nlprun -n train-mms-ctc-aleb-0.0001-1 -g 1 -a asrdro -o results/exp051/train-mms-ctc-aleb-0.0001-1.txt --mail-user ananjan -d a6000 'source ../../../tools/activate_python.sh; make train_mms_ctc_aleb_0.0001_1' -p high
nlprun -n train-mms-ctc-aleb-0.0001-2 -g 1 -a asrdro -o results/exp051/train-mms-ctc-aleb-0.0001-2.txt --mail-user ananjan -d a6000 'source ../../../tools/activate_python.sh; make train_mms_ctc_aleb_0.0001_2' -p high

nlprun -n train-xlsr-ctc-aleb-0.0001 -g 1 -a asrdro -o results/exp054/train-xlsr-ctc-aleb-0.0001.txt --mail-user ananjan -d a6000 'source ../../../tools/activate_python.sh; make train_xlsr_ctc_aleb_0.0001' -p high

nlprun -n train-mms-ctc-aleb-0.0001 -g 1 -a asrdro -o results/exp064/train-mms-ctc-aleb-0.0001.txt --mail-user ananjan -d a6000 'source ../../../tools/activate_python.sh; make train_mms_ctc_aleb_0.0001' -p high