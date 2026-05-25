# ============================================================
# cochran_q.py
# Test Q de Cochran + post-hoc McNemar
#
# Objectif : évaluer si le taux de recommandations dangereuses
# (reco_dangereuse) et d'informations inventées (info_inventee)
# varie selon le type de reformulation du prompt.
#
# Auteur : Julia CANARELLI — ESME Paris ING2 2025-2026
# Projet : LLMQ — Évaluation des performances d'un LLM
#          en contexte d'aide au diagnostic clinique
# ============================================================

from statsmodels.stats.contingency_tables import cochrans_q, mcnemar
import numpy as np

# ============================================================
# DONNÉES — remplacer les x par les valeurs réelles (0 ou 1)
# 0 = absence  |  1 = présence
# Une valeur par cas clinique (15 cas)
# ============================================================

# Variable : reco_dangereuse (recommandation dangereuse)
reco_narratif  = [x, x, x, x, x, x, x, x, x, x, x, x, x, x, x]
reco_condense  = [x, x, x, x, x, x, x, x, x, x, x, x, x, x, x]
reco_reformule = [x, x, x, x, x, x, x, x, x, x, x, x, x, x, x]

# Variable : info_inventee (hallucination additive)
hall_narratif  = [x, x, x, x, x, x, x, x, x, x, x, x, x, x, x]
hall_condense  = [x, x, x, x, x, x, x, x, x, x, x, x, x, x, x]
hall_reformule = [x, x, x, x, x, x, x, x, x, x, x, x, x, x, x]


def run_cochran(var_name, narratif, condense, reformule):
    """Lance le test Q de Cochran et les post-hoc McNemar."""

    print("=" * 60)
    print(f"TEST Q DE COCHRAN — {var_name}")
    print("=" * 60)

    data = np.array([narratif, condense, reformule]).T
    result = cochrans_q(data)

    print(f"Statistique Q : {result.statistic:.4f}")
    print(f"p-value        : {result.pvalue:.6f}")
    print()

    if result.pvalue < 0.05:
        print("✓ Résultat significatif (p < 0.05)")
        print("  → La reformulation influence significativement")
        print(f"    le taux de {var_name}.")
    else:
        print("✗ Résultat non significatif (p ≥ 0.05)")
        print(f"  → Le taux de {var_name} est stable")
        print("    quelle que soit la reformulation.")

    print()
    print("POST-HOC McNemar :")

    def mcnemar_pair(x, y, label):
        table = [[0, 0], [0, 0]]
        for a, b in zip(x, y):
            table[a][b] += 1
        res = mcnemar(table, exact=True)
        sig = "✓" if res.pvalue < 0.05 else "✗"
        print(f"  {label:30s} : p = {res.pvalue:.6f}  {sig}")

    mcnemar_pair(narratif, condense,  "Narratif vs Condensé")
    mcnemar_pair(narratif, reformule, "Narratif vs Reformulé")
    mcnemar_pair(condense, reformule, "Condensé vs Reformulé")
    print()


# ============================================================
# EXÉCUTION
# ============================================================

run_cochran(
    "Recommandations dangereuses (reco_dangereuse)",
    reco_narratif, reco_condense, reco_reformule
)

run_cochran(
    "Informations inventées (info_inventee)",
    hall_narratif, hall_condense, hall_reformule
)
