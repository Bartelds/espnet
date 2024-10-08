# Usage: python local/plot_qs.py --log_pth exp/_exp_008/asr_train_asr_xlsr_sceb_dro_0.001_uniforminit/train.log

import argparse
import os
import re
import matplotlib.pyplot as plt
import numpy as np

def parse_log_file(file_path):
    data = []
    current_q_values = {}
    with open(file_path, 'r') as file:
        for line in file:
            if "normalized dro_q:" in line:
                # If current_q_values is not empty, append it to data
                if current_q_values:
                    data.append(current_q_values)
                current_q_values = {}
            elif "q[group#" in line:
                match = re.search(r'q\[group#([^\]]+)\]=\s*([\d.eE+-]+)', line)
                if match:
                    group = match.group(1)
                    q_value = float(match.group(2))
                    current_q_values[group] = q_value
        # After file read, append the last q_values if any
        if current_q_values:
            data.append(current_q_values)
    return data

def plot_data(data, save_dir):
    # data is a list of dicts
    groups = set()
    for q_values in data:
        groups.update(q_values.keys())
    
    group_q_values = {}
    for group in groups:
        group_q_values[group] = []
    
    for q_values in data:
        for group in groups:
            if group in q_values:
                group_q_values[group].append(q_values[group])
            else:
                group_q_values[group].append(None)

    # Write parsed group_q_values to a file
    # output_file_path = os.path.join(save_dir, f"q_values_{os.path.basename(save_dir)}.txt")
    # with open(output_file_path, 'w') as f:
    #     for group, q_values in group_q_values.items():
    #         f.write(f"Group {group} q-values:\n")
    #         for value in q_values:
    #             f.write(f"{value}\n")
    #         f.write("\n")
    
    plt.figure(figsize=(10, 6))
    color_map = plt.get_cmap('rainbow')
    colors = color_map(np.linspace(0, 1, len(group_q_values)))
    
    for i, (group, q_values) in enumerate(group_q_values.items()):
        plt.plot(q_values, label=group, color=colors[i])
    
    plt.xlabel('Iteration')
    plt.ylabel('q Value')
    plt.title('q Values Across Iterations')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"q_values_{os.path.basename(save_dir)}.png")
    # plt.xlim(0, 100)
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot q values from log file")
    parser.add_argument('--log_pth', type=str, help='The path to the log file')
    args = parser.parse_args()
    
    log_file_path = args.log_pth
    save_dir = os.path.dirname(log_file_path)
    data = parse_log_file(log_file_path)
    plot_data(data, save_dir)

if __name__ == '__main__':
    main()
