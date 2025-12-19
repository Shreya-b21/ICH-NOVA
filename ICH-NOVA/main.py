# main.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import pickle
import pandas as pd
from tqdm import trange
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# Import project modules
# ----------------------------
from src.generator.de_novo_generator import GraphDiffusionGenerator
from src.binding.gnn_binding import GNNBindingPredictor
from src.synthesis.synthesis_predictor import SynthesisPredictor
from src.stability.stability_zone_iv import StabilityPredictor
from src.admet.admet_predictor import admet_pass

# ----------------------------
# Load embeddings
# ----------------------------
with open("data/processed/embeddings.pkl", "rb") as f:
    embeddings_dict = pickle.load(f)

# ----------------------------
# Load models
# ----------------------------
generator = GraphDiffusionGenerator()
binding_model = GNNBindingPredictor()
synth_model = SynthesisPredictor()

stability_model = StabilityPredictor(input_dim=256, hidden_dim=128)
stability_model.load_state_dict(
    torch.load("models/stability_model.pt", map_location="cpu")
)
stability_model.eval()

# ----------------------------
# RL Hyperparameters
# ----------------------------
num_iterations = 10
num_molecules_per_iter = 30

alpha = 1.0   # binding
gamma = 0.3   # synthesis penalty
delta = 0.8   # stability

results = []

# ----------------------------
# RL Loop
# ----------------------------
for it in trange(num_iterations, desc="RL Iterations"):
    smiles_list = generator.generate(
        "data/protein_target.fasta",
        num_molecules=num_molecules_per_iter
    )

    binding_scores = binding_model.predict(smiles_list)
    synthesis_scores = synth_model.score(smiles_list)

    for smi in smiles_list:
        admet_ok = admet_pass(smi)

        if not admet_ok:
            results.append({
                "iteration": it + 1,
                "smiles": smi,
                "binding_score": None,
                "synthesis_score": None,
                "stability_pred": None,
                "reward": None,
                "admet_pass": 0
            })
            continue

        emb = embeddings_dict.get(smi, torch.randn(256)).unsqueeze(0)

        with torch.no_grad():
            stability_pred = stability_model(emb).item()

        binding = binding_scores[smi]
        synthesis = synthesis_scores[smi]

        reward = (
            alpha * binding
            - gamma * synthesis
            + delta * stability_pred
        )

        results.append({
            "iteration": it + 1,
            "smiles": smi,
            "binding_score": binding,
            "synthesis_score": synthesis,
            "stability_pred": stability_pred,
            "reward": reward,
            "admet_pass": 1
        })

# ----------------------------
# Save results
# ----------------------------
final_df = pd.DataFrame(results)
final_df.to_csv("final_candidates_rl.csv", index=False)

print("✅ Saved final_candidates_rl.csv")

# ----------------------------
# Plots (optional, local only)
# ----------------------------
plt.figure(figsize=(8,5))
mean_reward = final_df.groupby("iteration")["reward"].mean()
plt.plot(mean_reward.index, mean_reward.values, marker="o")
plt.xlabel("Iteration")
plt.ylabel("Average Reward")
plt.title("RL Reward Progression")
plt.grid(True)
plt.show()
