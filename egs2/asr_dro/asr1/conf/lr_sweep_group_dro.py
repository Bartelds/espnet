import yaml

smoothing = [0.0]
step_size = [1e-5, 1e-4, 1e-3, 1e-2]

def sweep_files(file_path):
    for s in smoothing:
        for ss in step_size:
            with open(f'{file_path}.yaml', 'r') as f:
                config = yaml.safe_load(f)
                
            config['num_iters_per_epoch'] = 1200
            config['ctc_conf']['dro_step_size'] = ss
            config['ctc_conf']['dro_q_epsilon'] = 1e-10
            config['ctc_conf']['accumulation'] = False
            config['ctc_conf']['max_epoch'] = 40
            config['ctc_conf']['smoothing'] = s
            config['ctc_conf']['agg'] = "mean"
            config['ctc_conf']['normalize_grad'] = False
            config['ctc_conf']['accumulation'] = False
            config['optim_conf']['lr'] = 1e-4
            config['optim_conf']['weight_decay'] = 1e-6
            config['max_epoch'] = 40
            config['accum_grad'] = 16
            config['batch_size'] = 4
            config['keep_nbest_models'] = 2

            with open(f'{file_path}_{ss}_base.yaml', 'w') as f:
                yaml.dump(config, f)

if __name__=='__main__':
    sweep_files('mms_example_group_dro')
    sweep_files('xlsr_example_group_dro')
