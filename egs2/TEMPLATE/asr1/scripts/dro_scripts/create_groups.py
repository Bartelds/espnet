import argparse

def main(args):
    out_list = []
    with open(args.utt2spk_file, 'r') as f:
        for line in f:
            line = line.strip().split()
            out_list.append((line[0], line[1].split('_')[1]))

    with open(args.out_utt2category_file, 'w') as f:
        for _ in out_list:
            f.write(f'{_[0]} {_[1]}\n')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--utt2spk-file', required=True)
    parser.add_argument('--out-utt2category-file', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)