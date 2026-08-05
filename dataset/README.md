Dataset format (JSONL)
- Each line is a JSON object representing one labeled example (context + target + definition/concept)
- Fields (recommended):
  - id: unique id
  - target: the surface form of the target word (string)
  - target_span: [start_char, end_char] relative to context (optional; helps locate the token)
  - sentence: full context sentence (string)
  - gloss: human-readable definition or paraphrase that matches the sense of the target in this context (string)
  - concept_id: optional canonical ID for the sense/concept (e.g., wordnet:wn:01234567, wikidata:Qxxx)
  - paraphrases: optional list of paraphrases (strings)
  - modality: optional dict with other grounding (image_url, audio_url, knowledge_graph_triples)
  - source: where example came from (e.g., semcor, wiktionary, human)
  - lang: language code

Example line (JSON):
{"id":"ex-0001","target":"bank","target_span":[10,14],"sentence":"She walked along the river bank to clear her mind.","gloss":"the land alongside a river","concept_id":"wordnet:bank%1:17:00::","paraphrases":["riverside","river bank"],"source":"wiktionary","lang":"en"}

Data sources to consider
- WordNet + SemCor (sense-annotated corpora)
- Wiktionary example sentences & definitions
- Open Multilingual WordNet, BabelNet
- Wikidata labels and descriptions
- Paraphrase corpora (PPDB, ParaNMT)
- Web-curated contexts with weak labels (distant supervision) — be careful about noise

Data balancing
- Ensure senses are balanced; rare senses may need augmentation
- Consider multiple positive glosses per context (if available) and hard negatives (other glosses of the same word)
