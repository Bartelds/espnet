import yaml

LR = [5e-4,1e-4,5e-5,1e-5]

def sweep_files(file_path):
    for lr in LR:
        with open(f'{file_path}.yaml', 'r') as f:
            config = yaml.safe_load(f)
        config['optim_conf']['lr'] = lr
        with open(f'{file_path}_{float(lr)}.yaml', 'w') as f:
            yaml.dump(config, f)

if __name__=='__main__':
    sweep_files('train_asr_mms')
    sweep_files('train_asr_xlsr')