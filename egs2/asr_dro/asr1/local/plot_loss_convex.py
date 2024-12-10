import os
import re
import matplotlib.pyplot as plt
import numpy as np

def parse_baseline_exp(root_dir, baseline_exp):
    """ Parse baseline experiments that have logs with an epoch-end format """

    epoch_line_regex = re.compile(r"(\d+)epoch results:.*?\[valid\].*? loss=([\d\.]+)")
    results = {}

    for exp in baseline_exp:
        exp_path = os.path.join(root_dir, exp)
        if not os.path.isdir(exp_path):
            continue
        results[exp] = {}

        for item in os.listdir(exp_path):
            if os.path.isdir(os.path.join(exp_path, item)) and ("asr_train-mms-ctc" in item or "asr_train-xlsr-ctc" in item):
                model_dir = os.path.join(exp_path, item)
                model_type = "mms" if "mms" in item else "xlsr"
                log_file = os.path.join(model_dir, "train.log")

                epoch_losses = []
                with open(log_file, "r") as f:
                    for line in f:
                        match = epoch_line_regex.search(line)
                        if match:
                            epoch = int(match.group(1))
                            val_loss = float(match.group(2))
                            epoch_losses.append((epoch, val_loss))
                results[exp][model_type] = epoch_losses
    return results


def parse_dro_exp(root_dir, dro_exp_mms, dro_exp_xlsr):
    """ Parse DRO experiment logs that contain per-language validation samples """

    results = {}
    epoch_end_regex = re.compile(r"(\d+)epoch results:")
    val_sample_regex = re.compile(r"Validation Sample \d+: Language = (\S+),.*? Loss = ([\d\.]+)")

    dro_paths = [dro_exp_mms, dro_exp_xlsr]

    for rel_path in dro_paths:
        full_path = os.path.join(root_dir, rel_path)
        if not os.path.isdir(full_path):
            continue

        path_parts = rel_path.split('/')
        exp_name = path_parts[0]  # e.g. "_exp_099"
        model_type = "mms" if "mms" in rel_path else "xlsr"

        if exp_name not in results:
            results[exp_name] = {}
        if model_type not in results[exp_name]:
            results[exp_name][model_type] = {}

        log_file = os.path.join(full_path, "train.log")
        if not os.path.isfile(log_file):
            continue

        current_epoch = 1
        language_losses = {}  # {language: [losses for this epoch]}
        
        with open(log_file, "r") as f:
            for line in f:                  
                end_match = epoch_end_regex.search(line)
                # End of epoch
                if end_match:
                    # Store previous epoch
                    if language_losses:
                        store_epoch_language_losses(results, exp_name, model_type, current_epoch, language_losses)
                    # Start a processing new epoch
                    current_epoch = int(end_match.group(1)) + 1
                    language_losses = {}

                # Process epoch
                val_match = val_sample_regex.search(line)
                if val_match:
                    language = val_match.group(1)
                    loss = float(val_match.group(2)) / 16.0  # divide by gradient accum

                    if language not in language_losses:
                        language_losses[language] = []
                    language_losses[language].append(loss)

    return results


def store_epoch_language_losses(results, exp, model_type, epoch, language_losses):
    """ Compute average loss per language and store it in results """

    for lang, losses in language_losses.items():
        avg_loss = np.mean(losses)
        if lang not in results[exp][model_type]:
            results[exp][model_type][lang] = []
        results[exp][model_type][lang].append((epoch, avg_loss))


def plot_results(root_dir, baseline_exp, dro_exp_mms, dro_exp_xlsr, baseline_res, dro_res):
    """ Plot results of baseline exps and dro exps on a single figure """
    asr1_dir = os.path.dirname(root_dir)
    plot_dir = os.path.join(asr1_dir, "plt")
    os.makedirs(plot_dir, exist_ok=True)

    all_exps = baseline_exp + [dro_exp_mms.split('/')[0], dro_exp_xlsr.split('/')[0]]
    exps_str = "_".join(sorted(set(all_exps)))
    plot_path = os.path.join(plot_dir, f"{exps_str}_validation_loss.png")

    dro_exp_name = dro_exp_mms.split('/')[0]

    plt.figure(figsize=(10, 6))

    # Plot baseline experiments
    for ex in baseline_res:
        for model_type, data in baseline_res[ex].items():
            epochs = [e for (e, l) in data]
            losses = [l for (e, l) in data]
            label_name = f"{ex.lstrip('_')}-{model_type}"
            plt.plot(epochs, losses, marker='o', label=label_name)

    # Plot DRO experiment results
    if dro_exp_name in dro_res:
        for model_type, lang_data in dro_res[dro_exp_name].items():
            for lang, data in lang_data.items():
                epochs = [e for (e, l) in data]
                losses = [l for (e, l) in data]
                label_name = f"{dro_exp_name.lstrip('_')}-{model_type}-{lang}"
                plt.plot(epochs, losses, marker='x', linestyle='--', label=label_name)

    plt.title(f"Validation Loss across Epochs\nExperiments: {', '.join(set(all_exps))}")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.legend(loc='best') 
    plt.grid(True)

    plt.savefig(plot_path)
    plt.close()

    print(f"Saved plot to {plot_path}...")


def main():
    root_dir = "/nlp/scr/bartelds/git/asr-dro/espnet/egs2/asr_dro/asr1/exp"

    baseline_exp = ["_exp_095", "_exp_096"] # these are monolingual experiments

    dro_exp_mms = "_exp_099/asr_train_asr_mms_aleb_dro_0.0001_la_0.1"
    dro_exp_xlsr = "_exp_099/asr_train_asr_xlsr_aleb_dro_0.0001_la_0.1"

    baseline_res = parse_baseline_exp(root_dir, baseline_exp)
    dro_res = parse_dro_exp(root_dir, dro_exp_mms, dro_exp_xlsr)

    plot_results(root_dir, baseline_exp, dro_exp_mms, dro_exp_xlsr, baseline_res, dro_res)

if __name__ == "__main__":
    main()
