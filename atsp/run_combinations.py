import os
import subprocess

# ckpt = ['radar_official_checkpoint', 'softmax_dim_2', 'softmax_dim_3', 'sinkhorn_k_iter_-2', 'sinkhorn_k_iter_2']
ckpt = ['ACTsinkhorn']
nodes = [100]
# p_vals = [0, 20, 40, 60, 80, 100]
p_vals = [100]
# k_vals = [-1, 1, -2, 2, -5, 5, -10, 10, -20, 20, -30, 30]
k_vals = [-100]

env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "2"  # 指定只对进程暴露 1 号 GPU

for ckpt_name in ckpt:
    for node in nodes:    
        for p in p_vals:
            for k in k_vals:
                print("="*50)
                print(f"Running combination: p={p}, k={k}, node={node}, ckpt={ckpt_name} on GPU 1")
                print("="*50)
                subprocess.run(["python", "test.py", "--p", str(p), "--k", str(k), "--node", str(node), "--ckpt", ckpt_name], env=env)
