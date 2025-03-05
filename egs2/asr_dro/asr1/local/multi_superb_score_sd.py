import argparse
import os
import re
import json
import collections

import pandas as pd
import numpy as np


def parse_train_file(train, encoding="utf-8"):
    datasets = []
    with open(os.path.join("./data", train, "text"), "r") as f:
        for line in f:
            datasets.append(line.split()[0].split("_")[0].lower())

    return list(set(sorted(datasets)))    


def parse_results_file(result, encoding="utf-8"):
    start_capture = False
    data = []
    cnt = 0
    with open(result, "r") as f:
        for line in f:
            if "SYSTEM SUMMARY PERCENTAGES by SPEAKER" in line:
                start_capture = True
                continue
            if start_capture and re.match(r"^\|\s+Median", line):
                cleaned_line = line.replace("|", "").strip()
                parts = re.split(r"\s{2,}", cleaned_line)
                if len(parts) == 9:
                    data.append(parts)
                break  # Stop after including the Median row

            if start_capture:
                if re.match(r"^\|\s+[a-zA-Z]", line):
                    cleaned_line = line.replace("|", "").strip().split()
                    if len(cleaned_line) == 9:
                        data.append(cleaned_line)

    columns = ["SPKR", "# Snt", "# Wrd", "Corr", "Sub", "Del", "Ins", "Err", "S.Err"]
    result = pd.DataFrame(data, columns=columns)

    numeric_cols = columns[1:]  # Exclude SPKR for conversion
    result[numeric_cols] = result[numeric_cols].apply(pd.to_numeric)

    return result


def compute_micro_sd(result_speaker, out_file):
    result_speaker["Language"] = result_speaker["SPKR"].apply(lambda x: x.split("_")[1])
    language_grouped = result_speaker.groupby("Language").agg({"# Wrd": "sum", "Weighted Err": "sum"})
    language_grouped["Sum/Avg"] = language_grouped["Weighted Err"] / language_grouped["# Wrd"]

    weights = language_grouped["# Wrd"]
    values = language_grouped["Sum/Avg"]

    language_grouped.to_csv(out_file)

    weighted_mean = np.average(values, weights=weights)

    weighted_variance = np.average((values - weighted_mean)**2, weights=weights)
    weighted_micro_std_dev = np.sqrt(weighted_variance)

    return weighted_mean, weighted_micro_std_dev


def compute_macro_sd(result_speaker, n_langs):
    result_speaker["Language"] = result_speaker["SPKR"].apply(lambda x: x.split("_")[1])
    result_speaker["Dataset"] = result_speaker["SPKR"].apply(lambda x: x.split("_")[0])
    
    language_grouped = result_speaker.groupby(["Language", "Dataset"]).agg({"# Wrd": "sum", "Weighted Err": "sum"})
    language_grouped["Sum/Avg"] = language_grouped["Weighted Err"] / language_grouped["# Wrd"]

    max_values = language_grouped.groupby("Language")["Sum/Avg"].idxmax()
    min_values = language_grouped.groupby("Language")["Sum/Avg"].idxmin()
    
    max_datasets = language_grouped.loc[max_values, ["Sum/Avg"]].rename(columns={"Sum/Avg": "Max Sum/Avg"}).reset_index()
    min_datasets = language_grouped.loc[min_values, ["Sum/Avg"]].rename(columns={"Sum/Avg": "Min Sum/Avg"}).reset_index()

    merged_ranges = pd.merge(max_datasets, min_datasets, on="Language", suffixes=('_Max', '_Min'))
    merged_ranges["Range"] = merged_ranges["Max Sum/Avg"] - merged_ranges["Min Sum/Avg"]
    top_n_languages_macro = merged_ranges.sort_values(by="Range", ascending=False).head(n_langs)

    # Compute the macro-average "Sum/Avg" for each language by averaging the "Sum/Avg" scores across its datasets
    language_macro_avg = language_grouped.groupby("Language")["Sum/Avg"].mean()
    
    languages_to_remove = ["dan", "lit", "tur", "srp", "vie", "kaz", "zul", "tsn", "epo", "frr", "tok", "umb", "bos", "ful", "ceb", "luo", "kea", "sun", "tso", "tos"]
    language_macro_avg = language_macro_avg[~language_macro_avg.index.isin(languages_to_remove)]

    print(f"language_macro_avg:\n{language_macro_avg}")
    print(f"Lowest CER:\n{language_macro_avg.idxmin()} {language_macro_avg.min():.2f}")
    print(f"Highest CER:\n{language_macro_avg.idxmax()} {language_macro_avg.max():.2f}")

    # Compute the overall macro-average across all languages
    overall_macro_avg = language_macro_avg.mean()
    
    # Compute the standard deviation of the macro-averages across languages
    overall_macro_std = language_macro_avg.std()
    
    return overall_macro_avg, overall_macro_std, top_n_languages_macro.round(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_file", type=str, help="Path to the results file to process")
    parser.add_argument("--out_file", type=str)
    args = parser.parse_args()

    result = args.result_file
    if not os.path.exists(result):
        raise RuntimeError(f"Cannot find result file at {result}")

    # Parse results and prepare data
    result = parse_results_file(result)
    result_speaker = result.iloc[:-4, :]

    result_speaker["Weighted Err"] = result_speaker["Err"] * result_speaker["# Wrd"]
    
    # Calculate macro average and standard deviation
    n_langs = 1 # number of languages to return for dataset range perf.
    overall_micro_avg, overall_micro_std = compute_micro_sd(result_speaker, args.out_file)
    overall_macro_avg, overall_macro_std, top_n_languages_macro = compute_macro_sd(result_speaker, n_langs)
    
    print(f"Micro Sum/Avg: {overall_micro_avg:.2f}")
    print(f"Macro Sum/Avg: {overall_macro_avg:.2f}")
    print(f"Standard Deviation Across Languages: {overall_macro_std:.2f}")
    print(f"Top {n_langs} languages with highest dataset range:\n{top_n_languages_macro}")
