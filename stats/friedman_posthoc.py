# ============================================================
# friedman_posthoc.py
# Test de Friedman + post-hoc Wilcoxon avec correction Bonferroni
#
# Objectif : évaluer si le type de reformulation influence
# significativement le score d'exactitude clinique (S_exact)
# et la confiance déclarée.
#
# Auteur : Julia CANARELLI — ESME Paris ING2 2025-2026
# Projet : LLMQ — Évaluation des performances d'un LLM
#          en contexte d'aide au diagnostic clinique
# ============================================================

from scipy.stats import friedmanchisquare, wilcoxon

# ============================================================
# DONNÉES — remplacer les x par les valeurs réelles
# S_exact par cas clinique selon la reformulation
# ============================================================

# Score d'exactitude S_exact = N_correct / N_ref
# Une valeur par cas clinique (15 cas)
narratif  = [x, x, x, x, x, x, x, x, x, x, x, x, x, x, x]
condense  = [x, x, x, x, x, x, x, x, x, x, x, x, x, x, x]
reformule = [x, x, x, x, x, x, x, x, x, x, x, x, x, x, x]

# ============================================================
# TEST DE FRIEDMAN
# ============================================================

print("=" * 60)
print("TEST DE FRIEDMAN — Score d'exactitude S_exact")
print("=" * 60)

stat, p = friedmanchisquare(narratif, condense, reformule)

print(f"Statistique χ²  : {stat:.4f}")
print(f"p-value          : {p:.6f}")
print()

if p < 0.05:
    print("✓ Résultat significatif (p < 0.05)")
    print("  → Le type de reformulation influence significativement")
    print("    le score d'exactitude clinique.")
else:
    print("✗ Résultat non significatif (p ≥ 0.05)")
    print("  → Aucun effet significatif de la reformulation.")

# ============================================================
# POST-HOC — Tests de Wilcoxon avec correction de Bonferroni
# Alpha corrigé = 0.05 / 3 = 0.0167
# ============================================================

print()
print("=" * 60)
print("POST-HOC — Tests de Wilcoxon (correction Bonferroni α = 0.017)")
print("=" * 60)

alpha_bonferroni = 0.05 / 3

stat1, p1 = wilcoxon(narratif, condense)
stat2, p2 = wilcoxon(narratif, reformule)
stat3, p3 = wilcoxon(condense, reformule)

comparaisons = [
    ("Narratif vs Condensé",  p1),
    ("Narratif vs Reformulé", p2),
    ("Condensé vs Reformulé", p3),
]

for nom, p_val in comparaisons:
    sig = "✓ SIGNIFICATIF" if p_val < alpha_bonferroni else "✗ non significatif"
    print(f"  {nom:30s} : p = {p_val:.6f}  {sig}")

print()
print("=" * 60)
print("TEST DE FRIEDMAN — Confiance déclarée")
print("=" * 60)

# ============================================================
# DONNÉES — confiance déclarée par le modèle
# ============================================================

confiance_narratif  = [x, x, x, x, x, x, x, x, x, x, x, x, x, x, x]
confiance_condense  = [x, x, x, x, x, x, x, x, x, x, x, x, x, x, x]
confiance_reformule = [x, x, x, x, x, x, x, x, x, x, x, x, x, x, x]

stat_c, p_c = friedmanchisquare(
    confiance_narratif,
    confiance_condense,
    confiance_reformule
)

print(f"Statistique χ²  : {stat_c:.4f}")
print(f"p-value          : {p_c:.6f}")
print()

if p_c < 0.05:
    print("✓ La reformulation influence la confiance déclarée.")
else:
    print("✗ La reformulation n'influence pas significativement")
    print("  la confiance déclarée.")
    print("  → Dissociation confiance / exactitude confirmée.")
