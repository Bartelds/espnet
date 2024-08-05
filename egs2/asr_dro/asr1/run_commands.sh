nlprun -n 1hr_20subset_mms_d.txt -g 1 -a asr-dro -o 1hr_20subset_mms_d.txt --mail-user ananjan -r 80G -d a6000 -p high 'make train-mms'

nlprun -n 1hr_20subset_mms_dro_1_d.txt -g 1 -a asr-dro -o 1hr_20subset_mms_dro_1_d.txt --mail-user ananjan -r 80G -d a6000 -p high 'make train-mms-dro'

nlprun -n 1hr_20subset_mms_dro_01_d.txt -g 1 -a asr-dro -o 1hr_20subset_mms_dro_01_d.txt --mail-user ananjan -r 80G -d a6000 'make train-mms-dro2'

nlprun -n 1hr_20subset_xlsr_d.txt -g 1 -a asr-dro -o 1hr_20subset_xlsr_d.txt --mail-user ananjan -r 80G -d a6000 -p high 'make train-xlsr'

nlprun -n 1hr_20subset_xlsr_dro_1_d.txt -g 1 -a asr-dro -o 1hr_20subset_xlsr_dro_1_d.txt --mail-user ananjan -r 80G -d a6000 -p high 'make train-xlsr-dro'

nlprun -n 1hr_20subset_xlsr_dro_01_d.txt -g 1 -a asr-dro -o 1hr_20subset_xlsr_dro_01_d.txt --mail-user ananjan -r 80G -d a6000 'make train-xlsr-dro2'