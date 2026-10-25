import pickle
import numpy as np

root = "./MMEB-eval_outputs/InternVL3-14B/"

with open(root + "ImageNet-1K_qry", "rb") as f:
    q, _ = pickle.load(f)

with open(root + "ImageNet-1K_tgt", "rb") as f:
    t, _ = pickle.load(f)

q = np.asarray(q, dtype=np.float32)
t = np.asarray(t, dtype=np.float32)

q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
t = t / np.maximum(np.linalg.norm(t, axis=-1, keepdims=True), 1e-12)

print("Shapes:", q.shape, t.shape)

# Similarity giữa các query
q_small = q[:200]
q_sim = q_small @ q_small.T
mask = ~np.eye(len(q_small), dtype=bool)

print("Query pairwise cosine:", q_sim[mask].mean())

# Similarity giữa các target
t_small = t[:200]
t_sim = t_small @ t_small.T
mask = ~np.eye(len(t_small), dtype=bool)

print("Target pairwise cosine:", t_sim[mask].mean())

# Cross-modal similarity
sim = q @ t.T
pred = sim.argmax(axis=1)

print("Q-T cosine mean:", sim.mean())
print("Q-T cosine std:", sim.std())
print("Unique predicted classes:", len(np.unique(pred)))