# ============================================================
# spearman_correlation.py
# Corrélation de Spearman : confiance déclarée vs exactitude réelle
#
# Objectif : évaluer si le niveau de confiance auto-déclaré
# par le modèle est corrélé à la qualité réelle de la réponse
# évaluée par l'expert clinicien (H4).
#
# Auteur : Julia CANARELLI — ESME Paris ING2 2025-2026
# Projet : LLMQ — Évaluation des performances d'un LLM
#          en contexte d'aide au diagnostic clinique
# ============================================================

from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# DONNÉES — remplacer par les valeurs réelles
# confidence : score de confiance auto-déclaré (0-100)
# accuracy   : score d'exactitude S_exact (0-1)
# ============================================================

confidence = [x, x, x, x, x, x, x, x, x, x, x, x, x, x, x]  # scores /100
accuracy   = [x, x, x, x, x, x, x, x, x, x, x, x, x, x, x]  # S_exact [0,1]

# ============================================================
# CORRÉLATION DE SPEARMAN
# ============================================================

print("=" * 60)
print("CORRÉLATION DE SPEARMAN")
print("Confiance déclarée vs Exactitude réelle (S_exact)")
print("=" * 60)

rho, p = spearmanr(confidence, accuracy)

print(f"ρ (rho)   : {rho:.3f}")
print(f"p-value   : {p:.5f}")
print()

if p < 0.05:
    print("✓ Corrélation statistiquement significative (p < 0.05)")
    if rho > 0.5:
        print("  Corrélation positive forte → bonne calibration")
    elif rho > 0:
        print("  Corrélation positive faible → calibration insuffisante")
        print("  → La confiance explique peu l'exactitude réelle.")
    else:
        print("  ⚠️  Corrélation négative → surconfiance inversée")
else:
    print("✗ Aucune corrélation significative (p ≥ 0.05)")
    print("  → La confiance déclarée est déconnectée de la qualité réelle.")

print()
print(f"Interprétation clinique :")
print(f"  Un ρ = {rho:.3f} signifie que la confiance affichée")
print(f"  explique très peu la justesse réelle des réponses.")
print(f"  → Ne pas utiliser la confiance déclarée comme")
print(f"    indicateur de fiabilité médicale.")

# ============================================================
# SCATTER PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(7, 5))

ax.scatter(
    [c / 100 for c in confidence],  # normaliser sur [0,1]
    accuracy,
    color="#C0392B",
    alpha=0.7,
    s=80,
    edgecolors="white",
    linewidths=0.5
)

# Ligne de calibration parfaite
ax.plot([0, 1], [0, 1], "b--", alpha=0.4, label="Calibration parfaite")

# Annotations
ax.set_xlabel("Confiance déclarée (normalisée)", fontsize=12)
ax.set_ylabel("Exactitude réelle (S_exact)", fontsize=12)
ax.set_title(
    f"Corrélation de Spearman : confiance vs exactitude\nρ = {rho:.3f}, p = {p:.5f}",
    fontsize=12
)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/spearman_correlation.png", dpi=150, bbox_inches="tight")
plt.show()

print()
print(" Graphique sauvegardé : results/spearman_correlation.png")
