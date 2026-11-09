# miniRAG  discute avec tes PDF

Bienvenue sur mon miniRAG

J'ai codé ce petit assistant pour réviser mes cours à partir de mes slides :
je lui pose mes questions et il répond en citant la page. Il peut aussi me générer des
questions à choix multiples pour préparer mes examens. Tout tourne en local.

## Comment ça marche

1. **Lecture** des PDF page par page (`pypdf`)
2. **Découpage** du texte en morceaux d'environ 800 caractères qui se chevauchent
3. **Embeddings** : chaque morceau devient un vecteur (`sentence-transformers`)
4. **Stockage** dans une base vectorielle locale (`Chroma`, dossier `chroma_db/`)
5. **Recherche** des morceaux les plus proches de la question
6. **Génération** de la réponse par un LLM local (`Ollama`), avec les sources citées

```
PDF ──> texte ──> morceaux ──> vecteurs ──> Chroma
                                              │
question ──> vecteur ──> recherche ───────────┘
                              │
                     passages pertinents + question ──> Ollama ──> réponse + sources
```

## Installation

1. Installer [Ollama](https://ollama.com) puis télécharger un modèle :

   ```bash
   ollama pull llama3.2
   ```

2. Installer les dépendances Python :

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

Au premier lancement, le modèle d'embeddings (~450 Mo) est téléchargé une seule fois,
ensuite tout fonctionne hors ligne.

## Utilisation

### Interface web

```bash
streamlit run app.py
```

Dépose tes PDF dans la barre latérale, clique sur « Indexer », puis pose tes questions.

### Ligne de commande

```bash
python cli.py add docs/mon_cours.pdf      # indexer
python cli.py ask "De quoi parle le chapitre 2 ?"
python cli.py chat                          # mode conversation
python cli.py list                          # documents indexés
python cli.py reset                         # vider la base
```

## Configuration

Variables d'environnement facultatives :

| Variable          | Défaut                                   |
|-------------------|------------------------------------------|
| `RAG_LLM_MODEL`   | `llama3.2`                               |
| `RAG_EMBED_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2`  |
| `OLLAMA_URL`      | `http://localhost:11434`                 |
| `RAG_DB_DIR`      | `chroma_db`                              |

Exemple : `RAG_LLM_MODEL=mistral streamlit run app.py`

## Structure

```
rag.py            # coeur : lecture, découpage, indexation, recherche, génération
app.py            # interface Streamlit
cli.py            # interface ligne de commande
requirements.txt
docs/             # tu peux y mettre tes PDF (ignorés par git)
```

## Limites

- Les PDF scannés (images) ne sont pas lus : il faudrait ajouter de l'OCR.
- La qualité des réponses dépend du modèle Ollama choisi.
