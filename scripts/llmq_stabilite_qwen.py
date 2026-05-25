# ============================================================
# llmq_stabilite_qwen.py
# Script LLMQ — Test de stabilité Qwen 2.5 32B uniquement
# 15 cas × 3 reformulations × 1 modèle × 3 répétitions = 135 requêtes
#
# Auteur : Matthieu ADNET — ESME Paris ING2 2025-2026
# Projet : LLMQ — Évaluation des performances d'un LLM
#          en contexte d'aide au diagnostic clinique
#
# Note : Timeout étendu à 600s car Qwen 2.5 32B déborde
# partiellement sur CPU (67% GPU / 33% CPU sur RX 7900 XTX)
# → temps de réponse de 44s à 4min15s par prompt.
# ============================================================

import requests
import json
import datetime
import re

# --- Configuration ---
OLLAMA_URL  = "http://localhost:11434/api/chat"
INPUT_FILE  = "prompts/prompts_llmq_clean.json"
OUTPUT_FILE = "results/resultats_llmq_qwen.json"

MODELS = [
    "qwen2.5:32b"
]

NB_REPETITIONS = 3

# --- Extraction du score de confiance ---
def extraire_score(reponse):
    """
    Extrait le score de confiance auto-déclaré par le modèle.
    Pattern attendu : 'SCORE DE CONFIANCE : X/100'
    Retourne None si le score n'est pas détecté.
    """
    match = re.search(r"SCORE DE CONFIANCE\s*:\s*(\d+)", reponse, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None

# --- Fonction d'envoi ---
def envoyer_prompt(model, contexte, demande):
    """
    Envoie une requête HTTP POST à l'API Ollama.
    Temperature = 0.7 pour observer les variations entre répétitions.
    Timeout = 600s pour accommoder la lenteur de Qwen 2.5 32B.
    """
    message = f"""{contexte}

{demande}

A la fin de ta reponse, indique obligatoirement sur une nouvelle ligne :
SCORE DE CONFIANCE : X/100
ou X est ton niveau de confiance dans ta reponse (0 = pas du tout sur, 100 = certain)."""

    try:
        response = requests.post(OLLAMA_URL, json={
            "model": model,
            "messages": [{"role": "user", "content": message}],
            "stream": False,
            "options": {"temperature": 0.7}
        }, timeout=600)  # 10 minutes max pour Qwen 32B

        reponse_texte = response.json()["message"]["content"]
        score = extraire_score(reponse_texte)
        return reponse_texte, score, "success"

    except Exception as e:
        return f"ERREUR: {str(e)}", None, "error"

# --- Chargement des prompts ---
print("Chargement des prompts...")
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

resultats = {}
total = len(data) * 3 * len(MODELS) * NB_REPETITIONS
compteur = 0

print(f"Total : {len(data)} cas x 3 reformulations x {len(MODELS)} modèles x {NB_REPETITIONS} répétitions = {total} requêtes")
print(f"Modèle testé : {MODELS[0]}")
print(f"  Temps estimé : ~4h30 (2 min/réponse en moyenne)\n")

# --- Boucle principale ---
for case_id, case_data in data.items():
    resultats[case_id] = {
        "reponse_attendue": case_data["reponse_attendue"],
        "prompts": {}
    }

    for prompt in case_data["prompts"]:
        prompt_id = prompt["prompt_id"]
        resultats[case_id]["prompts"][prompt_id] = {}

        for model in MODELS:
            resultats[case_id]["prompts"][prompt_id][model] = {
                "contexte": prompt["contexte"],
                "demande": prompt["demande"],
                "repetitions": [],
                "scores_confiance": [],
                "score_moyen": None
            }

            scores = []

            for rep in range(1, NB_REPETITIONS + 1):
                compteur += 1
                print(f"[{compteur}/{total}] {prompt_id} → {model} (rep {rep}/{NB_REPETITIONS})...")

                reponse, score, statut = envoyer_prompt(
                    model,
                    prompt["contexte"],
                    prompt["demande"]
                )

                resultats[case_id]["prompts"][prompt_id][model]["repetitions"].append({
                    "repetition": rep,
                    "reponse": reponse,
                    "score_confiance": score,
                    "statut": statut,
                    "timestamp": datetime.datetime.now().isoformat()
                })

                if score is not None:
                    scores.append(score)

                if statut == "success":
                    score_str = f"Score : {score}/100" if score else "Score : non détecté"
                    print(f"  ✓ ({len(reponse)} car.) — {score_str}")
                else:
                    print(f"  ✗ ERREUR")

                # Sauvegarde intermédiaire — critique pour Qwen
                # (évite de perdre les données si le script est interrompu)
                with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                    json.dump(resultats, f, ensure_ascii=False, indent=2)

            # Score moyen sur les répétitions
            if scores:
                score_moyen = round(sum(scores) / len(scores), 1)
                resultats[case_id]["prompts"][prompt_id][model]["scores_confiance"] = scores
                resultats[case_id]["prompts"][prompt_id][model]["score_moyen"] = score_moyen
                print(f"  → Score moyen : {score_moyen}/100\n")

# --- Sauvegarde finale ---
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(resultats, f, ensure_ascii=False, indent=2)

# --- Résumé final ---
print(f"\n{'='*60}")
print(f" Terminé !")
print(f" Résultats : {OUTPUT_FILE}")
print(f"\n Récapitulatif des scores moyens :")
for case_id, case_data in resultats.items():
    for prompt_id, prompt_data in case_data["prompts"].items():
        for model, model_data in prompt_data.items():
            score = model_data.get("score_moyen")
            if score:
                print(f"  {prompt_id} | {model:20s} | {score}/100")
print(f"{'='*60}")
