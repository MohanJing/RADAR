import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def resolve_npz_file(data_dir, filename=None):
	if filename is not None:
		npz_path = os.path.join(data_dir, filename)
		if not os.path.exists(npz_path):
			raise FileNotFoundError(f'File not found: {npz_path}')
		return npz_path

	npz_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])
	if not npz_files:
		raise FileNotFoundError(f'No .npz file found in: {data_dir}')
	return os.path.join(data_dir, npz_files[0])


def load_sample_matrix(npz_path, sample_idx):
	data_obj = np.load(npz_path)
	if 'data' not in data_obj:
		raise KeyError(f"'data' key not found in {npz_path}. Keys: {list(data_obj.keys())}")

	data = data_obj['data']
	if data.ndim != 3:
		raise ValueError(f'Expected 3D array (batch, cnt, cnt), got shape: {data.shape}')

	if sample_idx < 0 or sample_idx >= data.shape[0]:
		raise IndexError(f'sample_idx {sample_idx} out of range [0, {data.shape[0]-1}]')

	return data[sample_idx]


def visualize_distance_matrix(matrix, save_path, title):
	cnt = matrix.shape[0]
	plt.figure(figsize=(8, 7))
	sns.heatmap(
		matrix,
		cmap='coolwarm',
		square=True,
		cbar=True,
		xticklabels=False,
		yticklabels=False,
	)
	plt.title(title)
	plt.xlabel(f'Nodes (cnt={cnt})')
	plt.ylabel(f'Nodes (cnt={cnt})')
	plt.tight_layout()
	plt.savefig(save_path, dpi=300)
	plt.close()


def main():
	parser = argparse.ArgumentParser(
		description='Visualize one sample distance matrix (cnt x cnt) as a red-blue heatmap.'
	)
	parser.add_argument('--data_dir', type=str, required=True, help='Directory containing .npz dataset files')
	parser.add_argument('--sample_idx', type=int, required=True, help='Sample index (0-based)')
	parser.add_argument('--filename', type=str, default=None, help='Optional .npz filename in data_dir')
	parser.add_argument('--out', type=str, default=None, help='Optional output png path')
	args = parser.parse_args()

	npz_path = resolve_npz_file(args.data_dir, args.filename)
	matrix = load_sample_matrix(npz_path, args.sample_idx)

	if args.out is None:
		stem = os.path.splitext(os.path.basename(npz_path))[0]
		out_name = f'{stem}_sample{args.sample_idx}_matrix.png'
		out_path = os.path.join(os.path.dirname(__file__), out_name)
	else:
		out_path = args.out

	title = f'Distance Matrix Heatmap | {os.path.basename(npz_path)} | sample={args.sample_idx}'
	visualize_distance_matrix(matrix, out_path, title)
	print(f'Saved heatmap to: {out_path}')


if __name__ == '__main__':
	main()
