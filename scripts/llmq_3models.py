# ============================================================
# llmq_3models.py
# Script LLMQ — Test des 3 modèles (temperature = 0)
# 15 cas × 3 reformulations × 3 modèles = 135 requêtes
#
# Auteur : Matthieu ADNET — ESME Paris ING2 2025-2026
# Projet : LLMQ — Évaluation des performances d'un LLM
#          en contexte d'aide au diagnostic clinique
# ============================================================

import requests
import json
import datetime

# --- Configuration ---
OLLAMA_URL  = "http://localhost:11434/api/chat"
INPUT_FILE  = "prompts/prompts_llmq_clean.json"
OUTPUT_FILE = "results/resultats_llmq_3models.json"

MODELS = [
    "llama3.1:8b",
    "gemma3:27b",
    "qwen2.5:32b"
]

# --- Fonction d'envoi ---
def envoyer_prompt(model, contexte, demande):
    """
    Envoie une requête HTTP POST à l'API Ollama.
    Temperature = 0 pour garantir la reproductibilité déterministe.
    Retourne le texte de la réponse et le statut.
    """
    message = f"{contexte}\n\n{demande}"
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": model,
            "messages": [{"role": "user", "content": message}],
            "stream": False,
            "options": {"temperature": 0}
        }, timeout=600)
        return response.json()["message"]["content"], "success"
    except Exception as e:
        return f"ERREUR: {str(e)}", "error"

# --- Chargement des prompts ---
print("Chargement des prompts...")
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

resultats = {}
total = len(data) * 3 * len(MODELS)
compteur = 0

print(f"Total : {len(data)} cas x 3 reformulations x {len(MODELS)} modèles = {total} requêtes\n")

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
            compteur += 1
            print(f"[{compteur}/{total}] {prompt_id} → {model}...")

            reponse, statut = envoyer_prompt(
                model,
                prompt["contexte"],
                prompt["demande"]
            )

            resultats[case_id]["prompts"][prompt_id][model] = {
                "reponse": reponse,
                "statut": statut,
                "timestamp": datetime.datetime.now().isoformat()
            }

            if statut == "success":
                print(f"  ✓ OK ({len(reponse)} caractères)")
            else:
                print(f"  ✗ ERREUR")

            # Sauvegarde intermédiaire après chaque réponse
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(resultats, f, ensure_ascii=False, indent=2)

print(f"\n{'='*50}")
print(f" Terminé !")
print(f" Résultats : {OUTPUT_FILE}")
print(f"{'='*50}")
