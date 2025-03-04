import argparse
import os
import re
import json
import collections

import pandas as pd
pd.options.mode.copy_on_write = True
pd.set_option('display.max_rows', 500)
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
                # Clean and split the Median line to extract data
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

    # Convert the captured data into a DataFrame
    columns = ["SPKR", "# Snt", "# Wrd", "Corr", "Sub", "Del", "Ins", "Err", "S.Err"]
    result = pd.DataFrame(data, columns=columns)

    # Convert numeric columns
    numeric_cols = columns[1:]  # Exclude SPKR for conversion
    result[numeric_cols] = result[numeric_cols].apply(pd.to_numeric)

    return result


def compute_micro_sd(result_speaker, out_file):
    # Group by language and calculate total words and weighted error
    result_speaker["Language"] = result_speaker["SPKR"].apply(lambda x: x.split("_")[1])
    language_grouped = result_speaker.groupby("Language").agg({"# Wrd": "sum", "Weighted Err": "sum"})
    language_grouped["Sum/Avg"] = language_grouped["Weighted Err"] / language_grouped["# Wrd"]

    # Verbose calulation
    weights = language_grouped["# Wrd"]
    values = language_grouped["Sum/Avg"]

    # print(language_grouped)
    # print(weights)
    # print(values)
    language_grouped.to_csv(out_file)

    # To reconstruct the sum/avg score of the test set from the sum/avg scores per language we need to compute:
    # Weighted mean
    weighted_mean = np.average(values, weights=weights)

    # Similarly, we can compute the weighted standard deviation
    weighted_variance = np.average((values - weighted_mean)**2, weights=weights)
    weighted_micro_std_dev = np.sqrt(weighted_variance)

    return weighted_mean, weighted_micro_std_dev


def compute_macro_sd(result_speaker, n_langs):
    result_speaker["Language"] = result_speaker["SPKR"].apply(lambda x: x.split("_")[1])
    result_speaker["Dataset"] = result_speaker["SPKR"].apply(lambda x: x.split("_")[0])
    
    language_grouped = result_speaker.groupby(["Language", "Dataset"]).agg({"# Wrd": "sum", "Weighted Err": "sum"})
    language_grouped["Sum/Avg"] = language_grouped["Weighted Err"] / language_grouped["# Wrd"]

    # Randomly select languages
    # select_langs(language_grouped)

    # Compute the max and min "Sum/Avg" scores for each language and their corresponding datasets
    max_values = language_grouped.groupby("Language")["Sum/Avg"].idxmax()
    min_values = language_grouped.groupby("Language")["Sum/Avg"].idxmin()
    
    # Extract dataset names and "Sum/Avg" for min and max
    max_datasets = language_grouped.loc[max_values, ["Sum/Avg"]].rename(columns={"Sum/Avg": "Max Sum/Avg"}).reset_index()
    min_datasets = language_grouped.loc[min_values, ["Sum/Avg"]].rename(columns={"Sum/Avg": "Min Sum/Avg"}).reset_index()

    # Combine and select top N
    merged_ranges = pd.merge(max_datasets, min_datasets, on="Language", suffixes=('_Max', '_Min'))
    merged_ranges["Range"] = merged_ranges["Max Sum/Avg"] - merged_ranges["Min Sum/Avg"]
    top_n_languages_macro = merged_ranges.sort_values(by="Range", ascending=False).head(n_langs)

    # Compute the macro-average "Sum/Avg" for each language by averaging the "Sum/Avg" scores across its datasets
    language_macro_avg = language_grouped.groupby("Language")["Sum/Avg"].mean()
    
    languages_to_remove = ["dan", "lit", "tur", "srp", "vie", "kaz", "zul", "tsn", "epo", "frr", "tok", "umb", "bos", "ful", "ceb", "luo", "kea", "sun", "tso", "tos"]
    # Remove the rows with these languages
    language_macro_avg = language_macro_avg[~language_macro_avg.index.isin(languages_to_remove)]

    # Print
    print(f"language_macro_avg:\n{language_macro_avg}")
    print(f"Lowest CER:\n{language_macro_avg.idxmin()} {language_macro_avg.min():.2f}")
    print(f"Highest CER:\n{language_macro_avg.idxmax()} {language_macro_avg.max():.2f}")

    # Compute the overall macro-average across all languages
    overall_macro_avg = language_macro_avg.mean()
    
    # Compute the standard deviation of the macro-averages across languages
    overall_macro_std = language_macro_avg.std()
    
    return overall_macro_avg, overall_macro_std, top_n_languages_macro.round(1)


def select_langs(language_grouped):
    # ./local/score_macro.sh --exp_dir /nlp/scr/bartelds/git/espnet/egs2/ml_superb/asr1/exp/is24_results_redecode/finetuning/ctc/9-14/mms/trained

    flat_df = language_grouped.reset_index()
    sorted_df = flat_df.sort_values('Sum/Avg')

    low10 = int(len(sorted_df) * 0.10)
    top10_start = int(len(sorted_df) * 0.9)

    lowest_10p_entries = sorted_df.iloc[:low10]
    middle_80p_entries = sorted_df.iloc[low10:top10_start]
    top_10p_entries = sorted_df.iloc[top10_start:]

    def select_unique_languages(entries, num_languages=6, excluded_languages=set()):
        # np.random.seed(90876) # first set of reported exps
        # np.random.seed(92378)
        np.random.seed(892736) # new 6-language subs: ces, cmn, nan, pol, ron, spa
        # np.random.seed(817657) # new 6-language subs: lav, fin, swe, nld, kab, urd
        selected_languages = set()
        selected_entries = []

        while len(selected_languages) < num_languages and not entries.empty:
            selected_entry = entries.sample(n=1)
            language = selected_entry['Language'].values[0]
            if language not in selected_languages and language not in excluded_languages:
                selected_languages.add(language)
                selected_entries.append(selected_entry)
                entries = entries[entries['Language'] != language]  # Remove all entries of the selected language

        return pd.concat(selected_entries), selected_languages

    selected_languages = set()
    selected_bottom_10, langs_bottom = select_unique_languages(lowest_10p_entries, excluded_languages=selected_languages)
    selected_languages.update(langs_bottom)

    selected_middle_80, langs_middle = select_unique_languages(middle_80p_entries, excluded_languages=selected_languages)
    selected_languages.update(langs_middle)

    selected_top_10, langs_top = select_unique_languages(top_10p_entries, excluded_languages=selected_languages)

    print(f"Selected 10 unique language-dataset pairs from the bottom 10%:\n{selected_bottom_10}")
    print(f"\nSelected 10 unique language-dataset pairs from the middle 80%:\n{selected_middle_80}")
    print(f"\nSelected 10 unique language-dataset pairs from the top 10%:\n{selected_top_10}")

    final_bottom_25 = selected_bottom_10.sample(n=2)
    final_middle_50 = selected_middle_80.sample(n=2)
    final_top_25 = selected_top_10.sample(n=2)

    print(f"Selected 2 unique language-dataset pairs from the bottom 10%:\n{final_bottom_25}")
    print(f"\nSelected 2 unique language-dataset pairs from the middle 80%:\n{final_middle_50}")
    print(f"\nSelected 2 unique language-dataset pairs from the top 10%:\n{final_top_25}")


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
    # print(result)

    result_speaker["Weighted Err"] = result_speaker["Err"] * result_speaker["# Wrd"]
    
    # Calculate macro average and standard deviation
    n_langs = 1 # number of languages to return for dataset range perf.
    overall_micro_avg, overall_micro_std = compute_micro_sd(result_speaker, args.out_file)
    overall_macro_avg, overall_macro_std, top_n_languages_macro = compute_macro_sd(result_speaker, n_langs)

    # assert (
    #         result.iloc[-4]["Err"] == round(overall_micro_avg, 1)
    #     ), f"Sum/Avg from result.txt does not match computed Sum/Avg"
    
    print(f"Micro Sum/Avg: {overall_micro_avg:.2f}")
    print(f"Macro Sum/Avg: {overall_macro_avg:.2f}")
    print(f"Standard Deviation Across Languages: {overall_macro_std:.2f}")
    print(f"Top {n_langs} languages with highest dataset range:\n{top_n_languages_macro}")
