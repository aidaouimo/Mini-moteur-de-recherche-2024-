# 🔍 Mini Search Engine

> Moteur de recherche full-text en Python — Index inversé · TF-IDF · BM25 · Requêtes booléennes · Zéro dépendance externe

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Zero deps](https://img.shields.io/badge/deps-stdlib_only-22c55e?style=flat-square)
![Tests](https://img.shields.io/badge/tests-pytest-f59e0b?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)

---

## 📌 Présentation

Ce projet implémente un **moteur de recherche full-text minimaliste** entièrement en Python pur (stdlib uniquement). Il couvre l'ensemble de la chaîne IR (*Information Retrieval*) :

| Étape | Composant |
|---|---|
| Prétraitement du texte | `Tokenizer` — normalisation, stop-words, stemmer Porter |
| Structure d'indexation | `InvertedIndex` — index inversé positionnel |
| Modèle de pondération | `TFIDFRanker` — TF-IDF classique + BM25 (Okapi) |
| Analyse de requête | `QueryParser` — booléen (AND/OR/NOT) + phrases |
| Interface haut niveau | `SearchEngine` — façade tout-en-un |
| Benchmarks | `benchmarks/` — tests de montée en charge jusqu'à 50 000 docs |

---

## 🗂️ Structure du projet

```
mini-search-engine/
│
├── src/
│   ├── __init__.py
│   ├── tokenizer.py        # Pipeline NLP : normalisation, stop-words, stemmer Porter
│   ├── index.py            # Index inversé positionnel + sérialisation pickle/JSON
│   ├── ranker.py           # TF-IDF & BM25, génération de snippets
│   ├── query_parser.py     # Parseur booléen + phrases, exécuteur de requêtes
│   └── engine.py           # Façade SearchEngine (indexation + recherche + stats)
│
├── tests/
│   └── test_engine.py      # Suite de tests unitaires (pytest)
│
├── benchmarks/
│   └── benchmark.py        # Build time, throughput, croissance du vocabulaire
│
├── data/                   # Corpus à indexer (gitignored)
├── results/                # Sorties & index sérialisés (gitignored)
│
├── main.py                 # Script de démonstration + mode interactif
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

**Prérequis** : Python 3.9+ · aucune dépendance externe pour le moteur (stdlib only)

```bash
# Cloner le dépôt
cd mini-search-engine

# (Optionnel) environnement virtuel
python -m venv .venv && source .venv/bin/activate

# Installer pytest pour les tests
pip install -r requirements.txt
```

---

## 🚀 Utilisation

### Démo rapide (corpus intégré, 10 documents)

```bash
python main.py
```

### Mode interactif

```bash
python main.py --interactive
```

```
  🔍 > inverted index search engine
  🔍 > "TF-IDF" AND python
  🔍 > machine learning NOT graph
  🔍 > quit
```

### Indexer ses propres fichiers texte

```bash
python main.py --files data/*.txt --mode bm25 --top-k 10
```

### Benchmarks de performance

```bash
python main.py --benchmark
```

### Options CLI

| Option | Défaut | Description |
|---|---|---|
| `--interactive` | off | Mode de requête interactif |
| `--benchmark` | off | Lance les benchmarks de performance |
| `--files GLOB` | — | Indexe des fichiers `.txt` personnalisés |
| `--mode` | `bm25` | `tfidf` ou `bm25` |
| `--top-k N` | `5` | Nombre de résultats par requête |
| `--no-stem` | off | Désactive le stemmer Porter |

---

## 🧠 Architecture technique

### 1. Pipeline de tokenisation (`tokenizer.py`)

```
Texte brut
  → lowercase + normalisation Unicode (NFD → ASCII)
  → suppression ponctuation
  → tokenisation par espaces
  → filtrage stop-words
  → stemming Porter
  → [liste de tokens normalisés]
```

Le stemmer de Porter est implémenté **from scratch** en 5 étapes de réduction :
`running → run`, `algorithms → algorithm`, `searching → search`.

---

### 2. Index inversé positionnel (`index.py`)

```python
{
  "search": {
    "doc_freq": 3,
    "postings": {
      "d02": {"tf": 4, "positions": [5, 12, 31, 44]},
      "d10": {"tf": 2, "positions": [8, 19]},
      ...
    }
  }
}
```

Les positions permettent la **recherche de phrases exactes** par vérification de la consécutivité des tokens.

**Complexité :**

| Opération | Complexité |
|---|---|
| Indexation d'un document | O(n) — n = nombre de tokens |
| Lookup d'un terme | O(1) — dictionnaire Python |
| Recherche booléenne AND | O(min postings) |
| Recherche de phrase | O(P × d × p) — P phrases, d docs candidats, p positions |

---

### 3. Pondération TF-IDF & BM25 (`ranker.py`)

**TF-IDF classique :**

$$\text{score}(t, d) = \frac{\text{tf}(t,d)}{|d|} \times \log\!\left(\frac{N}{1 + \text{df}(t)}\right) + 1$$

**BM25 (Okapi) :**

$$\text{score}(t, d) = \text{IDF}(t) \times \frac{\text{tf}(t,d) \cdot (k_1 + 1)}{\text{tf}(t,d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$

Paramètres par défaut : `k1 = 1.5`, `b = 0.75`.

---

### 4. Syntaxe des requêtes (`query_parser.py`)

| Type | Exemple | Comportement |
|---|---|---|
| Simple | `machine learning` | OR implicite sur tous les tokens |
| Phrase | `"inverted index"` | Correspondance positionnelle exacte |
| AND | `python AND search` | Les deux termes obligatoires |
| OR | `java OR python` | Au moins un des deux termes |
| NOT | `search NOT graph` | Exclut les documents contenant le terme |
| Mixte | `"inverted index" AND python NOT java` | Combinaison |

---

## 🧪 Tests

```bash
# Lancer la suite complète
pytest tests/ -v

# Avec couverture de code
pytest tests/ -v --tb=short
```

La suite couvre : tokenizer, stemmer, index inversé, requêtes booléennes, phrases, TF-IDF, BM25, intégration `SearchEngine`.

---

## 📊 Performances

Résultats sur corpus synthétique (MacBook Pro M2, Python 3.11) :

| Corpus | Build time | Throughput (BM25) |
|---|---|---|
| 1 000 docs | ~0.05s | ~4 000 req/s |
| 10 000 docs | ~0.4s | ~1 500 req/s |
| 50 000 docs | ~2.1s | ~600 req/s |

Lancez `python main.py --benchmark` pour reproduire ces mesures sur votre machine.

---

## 💻 Utilisation en bibliothèque

```python
from src.engine import SearchEngine
from src.index import Document

# Créer et configurer le moteur
engine = SearchEngine(mode="bm25", language="en", use_stemming=True)

# Indexer des documents
docs = [
    Document("d1", "Introduction to Python", "Python is a high-level programming language."),
    Document("d2", "Search Engines", "Inverted indexes power modern search engines."),
]
engine.index_documents(docs)

# Recherche simple
results = engine.search("python programming", top_k=5)
for r in results:
    print(f"[{r.rank}] {r.title} (score={r.score:.4f})")
    print(f"    {r.body_snippet}")

# Requête booléenne
results = engine.search("search AND index NOT database")

# Recherche de phrase exacte
results = engine.search('"inverted index"')

# Statistiques du moteur
print(engine.stats())

# Sauvegarder / charger l'index
engine.save_index("results/my_index.pkl")
engine.load_index("results/my_index.pkl")
```

---

## 📄 Licence

Ce projet est distribué sous licence **MIT**. Voir le fichier `LICENSE` pour plus de détails.

---

## 👤 Auteur

**Mohamed Aidaoui**  
Étudiant Ingénieur Informatique — ESIEE Paris / Université Gustave Eiffel  
[LinkedIn](https://www.linkedin.com/in/mohamed-aidaoui/) · [Portfolio](https://69b63f26831756000857e346--extraordinary-eclair-12e880.netlify.app/) · aidaoui31@gmail.com
