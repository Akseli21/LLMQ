# LLMQ — Évaluation des performances d'un LLM en contexte d'aide au diagnostic clinique

> **Analyse de l'impact de la reformulation du prompt**  
> Projet d'application LLMQ — Ingénieur 2 (2N/2NS) — ESME Paris — 2025-2026

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-Academic-green)](LICENSE)

---

## Auteurs

| Étudiant | Rôle |
|---|---|
| **Matthieu ADNET** | Déploiement technique, scripts d'expérimentation, pipeline Python |
| **Julia CANARELLI** | Protocole clinique, analyse statistique, BERTScore |

**Tuteur pédagogique :** M. Sami SOUIHI  
**Experts cliniciens :** Dr Jean CANARELLI · Interne Chjara-Maria TOMASI

---

## Résumé du projet

Ce projet évalue expérimentalement l'impact de la **reformulation du prompt** sur la qualité, la sécurité et la stabilité des réponses produites par trois modèles de langage déployés localement.

**Problématique centrale :**
> Dans quelle mesure la reformulation d'un même cas clinique influence-t-elle la qualité, la stabilité, la sécurité et la fiabilité des réponses produites par un grand modèle de langage dans un contexte d'aide à l'interprétation de résultats biologiques ?

**Résultats clés :**
- La reformulation influence significativement l'exactitude clinique (**p = 0,007**, test de Friedman)
- Surconfiance systématique des modèles (**ECE = 0,359**, **ρ = 0,233**)
- 0% de recommandations dangereuses pour Qwen 2.5 32B
- Gemma3 27B = modèle le plus stable sémantiquement (BERTScore)

---

## Modèles utilisés

| Modèle | Développeur | Paramètres | VRAM | Rôle |
|---|---|---|---|---|
| LLaMA 3.1 8B Instruct | Meta AI | 8,0 Md | ~5 Go | Référence |
| Gemma3 27B | Google DeepMind | 27,4 Md | ~17 Go | Intermédiaire |
| Qwen 2.5 32B | Alibaba Cloud | 32,8 Md | ~20 Go | Haute performance |

Tous déployés localement via **Ollama** sur AMD RX 7900 XTX (24 Go VRAM).

---

## Structure du dépôt

```
LLMQ/
│
├── README.md                          ← Ce fichier
├── requirements.txt                   ← Dépendances Python
├── LICENSE                            ← Licence académique
│
├── prompts/
│   └── prompts_llmq_clean.json        ← 15 cas cliniques × 3 reformulations
│
├── scripts/                           ← Scripts d'expérimentation (M. ADNET)
│   ├── llmq_3models.py                ← Test des 3 modèles (135 requêtes)
│   ├── llmq_stabilite.py              ← Stabilité LLaMA + Gemma (3 répétitions)
│   ├── llmq_stabilite_qwen.py         ← Stabilité Qwen (3 répétitions)
│   └── bertscore_analysis.py          ← Analyse BERTScore (J. CANARELLI)
│
├── stats/                             ← Scripts statistiques (J. CANARELLI)
│   ├── friedman_posthoc.py            ← Test de Friedman + post-hoc Wilcoxon
│   ├── cochran_q.py                   ← Test Q de Cochran + post-hoc McNemar
│   └── spearman_correlation.py        ← Corrélation de Spearman + scatter plot
│
└── results/
    └── .gitkeep                       ← Les fichiers JSON de résultats (ignorés par git)
```

---

## Installation

### Prérequis

- Python 3.11
- [Ollama](https://ollama.com/download) installé et lancé
- GPU avec ~5 Go VRAM minimum (pour LLaMA 3.1 8B)

### 1. Cloner le dépôt

```bash
git clone https://github.com/Akseli21/LLMQ.git
cd LLMQ
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Télécharger les modèles Ollama

```bash
ollama pull llama3.1:8b
ollama pull gemma3:27b
ollama pull qwen2.5:32b
```

### 4. Vérifier que le serveur Ollama tourne

```bash
ollama serve
ollama ps
```

---

## Utilisation

### Lancer les expérimentations

**Test des 3 modèles (135 requêtes, temperature = 0) :**
```bash
python scripts/llmq_3models.py
```

**Test de stabilité LLaMA + Gemma (90 requêtes, temperature = 0.7, 3 répétitions) :**
```bash
python scripts/llmq_stabilite.py
```

**Test de stabilité Qwen (135 requêtes, temperature = 0.7, 3 répétitions) :**
```bash
python scripts/llmq_stabilite_qwen.py
```

> Les résultats sont sauvegardés automatiquement dans `results/` après chaque réponse.  
> En cas d'interruption, les données déjà collectées sont conservées.

### Lancer les analyses statistiques

```bash
python stats/friedman_posthoc.py
python stats/cochran_q.py
python stats/spearman_correlation.py
```

### Lancer l'analyse BERTScore

```bash
python scripts/bertscore_analysis.py
```

---

## Protocole expérimental

### Structure des prompts

Chaque cas clinique est décliné en 3 reformulations selon la nomenclature **AAA.X** :

| Type | Description |
|---|---|
| **AAA.1** | Narratif — contexte clinique complet + demande standard |
| **AAA.2** | Condensé — mêmes informations sous forme synthétique |
| **AAA.3** | Reformulé — même contexte narratif + demande reformulée |

### Volume total

```
15 cas × 3 reformulations × 3 modèles × 3 répétitions = 405 requêtes
```

### Paramètres d'inférence

| Paramètre | Valeur | Justification |
|---|---|---|
| temperature | 0 (reproductibilité) / 0.7 (stabilité) | Isoler l'effet de la reformulation |
| num_gpu | 99 | Chargement maximal en VRAM |
| num_ctx | 4096 | Suffisant pour les prompts cliniques |
| timeout | 300-600 s | Adapté à la lenteur de Qwen 32B |

---

## Données

Les cas cliniques sont issus des annales du **Collège des enseignants d'endocrinologie, diabète et maladies métaboliques (2022)**.

Les données sont **synthétiques** (aucune donnée patient réelle) et conformes aux recommandations RGPD.

---

## Grille d'évaluation

Les réponses sont évaluées par des experts cliniciens selon les variables suivantes :

| Variable | Type | Description |
|---|---|---|
| S_exact | [0,1] | Score de fidélité clinique (N_correct / N_ref) |
| N_ajout | Entier | Éléments non justifiés (hallucinations additives) |
| reco_dangereuse | Binaire | Recommandation potentiellement dangereuse |
| info_inventee | Binaire | Information non justifiable par le prompt |
| confiance_declaree | [0,100] | Score auto-déclaré par le modèle |

---

## Références

- Vaswani et al. (2017). *Attention Is All You Need.* NeurIPS.
- Touvron et al. (2023). *LLaMA: Open and Efficient Foundation Language Models.* arXiv:2302.13971
- Hager et al. (2024). *Evaluation and mitigation of the limitations of LLMs in clinical decision-making.* Nature Medicine.
- Zhao et al. (2021). *Calibrate Before Use.* arXiv:2102.09690
- Guo et al. (2017). *On Calibration of Modern Neural Networks.* ICML.
- NIST (2023). *AI Risk Management Framework (AI RMF 1.0).*
- OMS (2021). *Ethics and governance of artificial intelligence for health.*

---

## Licence

Projet académique — ESME Paris 2025-2026.  
Usage personnel et éducatif uniquement.
