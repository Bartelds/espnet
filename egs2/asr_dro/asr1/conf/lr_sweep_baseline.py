import yaml

LR = [1e-4]

def sweep_files(file_path):
    for lr in LR:
        with open(f'{file_path}.yaml', 'r') as f:
            config = yaml.safe_load(f)
        config['num_iters_per_epoch'] = 1200
        config['accum_grad'] = 16
        config['max_epoch'] = 40
        config['batch_size'] = 4
        config['optim_conf']['lr'] = lr
        config['optim_conf']['weight_decay'] = 1e-6
        config['keep_nbest_models'] = 2
        with open(f'{file_path}_{lr}.yaml', 'w') as f:
            yaml.dump(config, f)

if __name__=='__main__':
    sweep_files('mms_example_baseline')
    # sweep_files('xlsr_example_baseline')
