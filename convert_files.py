from glob import glob
from os import path, makedirs
import subprocess
from p_tqdm import p_map
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", "-i", type=str, help="input directory of only audio files")
parser.add_argument("--output_dir", "-o", type=str, help="output directory")
parser.add_argument("--sample_rate", "-s", type=int, help="sample rate of output audio files")
args = parser.parse_args()

in_dir = path.normpath(args.input_dir)
if args.sample_rate is not None:
	sr = args.sample_rate
else:
	sr = 16000
if args.output_dir is not None:
	out_dir = path.normpath(args.output_dir)
else:
	out_dir = f"{in_dir}_{int(sr/1000)}k"

if not (sr and in_dir and out_dir):
	parser.print_help()
	exit()
assert in_dir != out_dir, 'input and output directories should not coincide'


def convert(file):
	new_filename = file.replace(in_dir, out_dir)
	makedirs(path.dirname(new_filename), exist_ok=True)
	subprocess.call(['ffmpeg', '-n', '-i', file, '-ac', '1', '-ar', str(sr), '-loglevel', '-8', new_filename])


if __name__ == "__main__":
	p_map(convert, glob(path.join(in_dir, '**', '*.wav'), recursive=True))
	subprocess.call(['stty', 'sane'])