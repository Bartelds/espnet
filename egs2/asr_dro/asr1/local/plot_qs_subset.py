# Usage:
# python local/plot_qs.py --log_pth <path_to_log_file> --languages lang1 lang2 lang3

import argparse
import os
import re
import matplotlib.pyplot as plt
import numpy as np

def parse_log_file(file_path):
    data = {}
    iteration = 0

    with open(file_path, 'r') as file:
        for line in file:
            if "normalized dro_q:" in line:
                iteration += 1
            elif "q[group#" in line:
                match = re.search(r'q\[group#(\w+)\]= ([\d.]+)', line)
                if match:
                    group = match.group(1)
                    q_value = float(match.group(2))
                    if group not in data:
                        data[group] = []
                    data[group].append(q_value)

    max_length = max(len(values) for values in data.values())
    for group, values in data.items():
        # Calculate the number of missing entries (iterations where the group was not present)
        missing_entries = max_length - len(values)
        # Prepend None for missing iterations
        data[group] = [None] * missing_entries + values
    return data

def plot_data(data, save_dir, languages=None):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(12, 6))  # Adjust the figure size as needed

    # If languages list is provided, filter data accordingly
    if languages:
        data_subset = {group: data[group] for group in languages if group in data}
    else:
        data_subset = data  # If no languages specified, plot all

    # Check if data_subset is empty
    if not data_subset:
        print("No data available for the specified languages.")
        return

    # Use a colormap suitable for the number of languages
    num_languages = len(data_subset)
    color_map = plt.get_cmap('tab10' if num_languages <= 10 else 'tab20')
    colors = color_map(np.linspace(0, 1, num_languages))

    for i, (group, q_values) in enumerate(data_subset.items()):
        plt.plot(q_values, label=group, color=colors[i % len(colors)])

    plt.xlabel('Iteration')
    plt.ylabel('q Value')
    plt.title('q Values Across Iterations')

    # Place the legend outside the plot
    plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0.,
        fontsize='small'
    )

    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'q_values_subset.png')
    plt.savefig(save_path, bbox_inches='tight')  # Ensure the legend is fully captured
    print(f"Plot saved to {save_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot q values from log file")
    parser.add_argument('--log_pth', type=str, required=True, help='The path to the log file')
    parser.add_argument(
        '--languages', nargs='+', default=None,
        help='List of languages (groups) to plot. If not specified, all languages are plotted.'
    )
    args = parser.parse_args()

    log_file_path = args.log_pth
    save_dir = os.path.dirname(log_file_path)
    data = parse_log_file(log_file_path)
    plot_data(data, save_dir, languages=args.languages)

if __name__ == '__main__':
    main()
