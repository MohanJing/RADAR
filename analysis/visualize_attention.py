import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def visualize_attention_single(model, logger, result_folder, layer_idx=0, batch_idx=0, head_idx=0):
    mha_module = model.encoder.layers[layer_idx].mixed_score_MHA

    if not hasattr(mha_module, 'last_raw_scores') or not hasattr(mha_module, 'last_attention_weights'):
        logger.warning('Could not find attention tracking attributes. Skipping visualization.')
        return

    raw_scores = mha_module.last_raw_scores[batch_idx, head_idx].numpy()
    attention_weights = mha_module.last_attention_weights[batch_idx, head_idx].numpy()
    softmax_dim2 = mha_module.last_softmax_dim2[batch_idx, head_idx].numpy()
    softmax_dim3 = mha_module.last_softmax_dim3[batch_idx, head_idx].numpy()

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    sns.heatmap(raw_scores, ax=axes[0, 0], cmap='coolwarm', robust=True)
    axes[0, 0].set_title(f'Before Normalization (Raw Scores)\nLayer {layer_idx}, Head {head_idx}')
    axes[0, 0].set_xlabel('Key (col_cnt)')
    axes[0, 0].set_ylabel('Query (row_cnt)')

    sns.heatmap(softmax_dim2, ax=axes[0, 1], cmap='coolwarm', robust=True)
    axes[0, 1].set_title(f'Softmax(dim=2) (Column-wise sum=1)\nLayer {layer_idx}, Head {head_idx}')
    axes[0, 1].set_xlabel('Key (col_cnt)')
    axes[0, 1].set_ylabel('Query (row_cnt)')

    sns.heatmap(softmax_dim3, ax=axes[1, 0], cmap='coolwarm', robust=True)
    axes[1, 0].set_title(f'Softmax(dim=3) (Row-wise sum=1)\nLayer {layer_idx}, Head {head_idx}')
    axes[1, 0].set_xlabel('Key (col_cnt)')
    axes[1, 0].set_ylabel('Query (row_cnt)')

    sns.heatmap(attention_weights, ax=axes[1, 1], cmap='coolwarm', robust=True)
    axes[1, 1].set_title(f'Sinkhorn Normalization (Dual Stochastic)\nLayer {layer_idx}, Head {head_idx}')
    axes[1, 1].set_xlabel('Key (col_cnt)')
    axes[1, 1].set_ylabel('Query (row_cnt)')

    plt.tight_layout()
    save_path = os.path.join(result_folder, f'attention_l{layer_idx}_h{head_idx}.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f'Attention heatmap saved successfully at: {save_path}')

def visualize_attention_all_heads_raw_for_samples(
    model,
    logger,
    result_folder,
    sample_indices,
    batch_start_idx,
    layer_idx=0,
    n_cols=4,
):
    mha_module = model.encoder.layers[layer_idx].mixed_score_MHA

    if not hasattr(mha_module, 'last_raw_scores'):
        logger.warning('Could not find raw attention scores. Skipping visualization.')
        return

    if sample_indices is None:
        return

    sample_indices = sorted(set(int(idx) for idx in sample_indices))
    raw_scores_all = mha_module.last_raw_scores
    batch_size = raw_scores_all.shape[0]
    head_num = raw_scores_all.shape[1]
    n_cols = max(1, min(n_cols, head_num))

    for sample_idx in sample_indices:
        local_idx = sample_idx - batch_start_idx
        if local_idx < 0 or local_idx >= batch_size:
            continue

        raw_scores = raw_scores_all[local_idx].numpy()
        n_rows = (head_num + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes = np.array(axes).reshape(n_rows, n_cols)

        for head_idx in range(head_num):
            row_idx = head_idx // n_cols
            col_idx = head_idx % n_cols
            ax = axes[row_idx, col_idx]
            sns.heatmap(raw_scores[head_idx], ax=ax, cmap='coolwarm', robust=True)
            ax.set_title(f'Sample {sample_idx} | Layer {layer_idx} | Head {head_idx}')
            ax.set_xlabel('Key (col_cnt)')
            ax.set_ylabel('Query (row_cnt)')

        for empty_idx in range(head_num, n_rows * n_cols):
            row_idx = empty_idx // n_cols
            col_idx = empty_idx % n_cols
            axes[row_idx, col_idx].axis('off')

        plt.tight_layout()
        save_path = os.path.join(result_folder, f'attention_raw_sample{sample_idx}_layer{layer_idx}.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f'Raw attention heatmap saved successfully at: {save_path}')