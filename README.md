# MeaningGround: train models to learn concept-level word meaning

Purpose
- This repository provides a scaffold to train models that learn the "actual meaning" of words (concept-level semantics) by aligning contextual usage with human-readable definitions, paraphrases, and knowledge-base concepts instead of relying only on raw token co-occurrence.

Key approaches
- Use gloss/context contrastive learning: align context embeddings (target word in sentence) with definition/paraphrase embeddings.
- Leverage sense inventories (WordNet, Wikidata), dictionary glosses, and paraphrase corpora.
- Optionally incorporate knowledge graph embeddings and multimodal grounding (images, audio).

Contents
- dataset/: dataset format and examples
- examples/train_contrastive.py: starter training script (PyTorch / sentence-transformers)
- requirements.txt: Python dependencies
- CONTRIBUTING.md, LICENSE

Quickstart (local)
1. Create a repo and add these files, or clone and enter project dir:
   git init
   git add .
   git commit -m "init: MeaningGround scaffold"

2. Create a virtual env and install:
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

3. Prepare dataset: put a JSONL file at data/train.jsonl using the dataset format in dataset/README.md

4. Run training (example):
   python examples/train_contrastive.py --train data/train.jsonl --epochs 3 --batch-size 32

Notes & next steps
- Collect high-quality gloss/context pairs from WordNet, Wiktionary, Wikidata, and curated corpora (SemCor).
- Consider human annotation for ambiguous senses and for sense alignment across languages.
- Evaluate using standard WSD datasets (SemEval, Senseval), word similarity datasets, and downstream tasks (NLI, QA) where conceptual understanding matters.

If you want, I can:
- Create this repository on GitHub (tell me owner/repo)
- Add CI, datasets, or run experiments on sample data
- Provide a Jupyter notebook walkthrough
