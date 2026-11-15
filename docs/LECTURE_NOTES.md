# Lecture notes relevant to this project

Condensed from the UCU NLP course lecture PDFs (kept outside the repository). Use this instead of re-reading the PDFs. Each bullet ends with how it applies here.

## L1 - Introduction to NLP
- Lifecycle is CRISP-DM: business understanding -> data understanding -> data preparation -> modeling -> evaluation -> deployment, with loops back. Define the problem and annotate data are explicit steps. -> AGENTS.md section 3.
- Challenges: internationalization (many phrasings of one intent), context, ambiguity ("mission bike directions" = place vs shop). -> UK queries; router must separate title-like vs description-like queries.
- Text cleaning menu (lowercase, punctuation, stopwords, stemming vs lemmatization, emoji, URLs, chat words) with the warning "do we always need it?". Stemming can produce non-words, lemmatization returns dictionary forms. -> normalize for BM25 only (UK lemmatization), keep raw text for dense embeddings.
- Approaches ladder: classical (TF-IDF + LR/NB/SVM), DL (embeddings + LSTM), pretrained transformers, prompting LLMs. -> justify LLM choices against simpler baselines where cheap.

## L2 - Evaluation in NLP
- Intrinsic (component, benchmark, fast, may not transfer) vs extrinsic (in the full system, closer to user value, costly, hard to attribute). -> per-component + end-to-end evals.
- Protocols: holdout, cross-validation, predefined benchmarks. -> dev/test split, tune on dev only.
- Accuracy misleads on imbalanced data; use the confusion matrix. Precision when false positives are costly, recall when misses are costly. -> constraint violations are precision-critical.
- F1 depends on which class is positive; MCC uses all four cells and is robust to imbalance (+1 perfect, 0 random, -1 inverse). -> clarify-vs-not decision.
- ROC-AUC for balanced data/symmetric costs; PR curves for rare positives. Log loss for probabilistic outputs.
- Multi-label/multi-class averaging: micro (samples equal), macro (classes equal), weighted. -> router macro-F1.
- IR metrics introduced, detailed in L3.
- Generation is hard to evaluate (many valid outputs). BLEU = clipped n-gram precision + brevity penalty, unreliable per sentence. ROUGE = recall-oriented (N, L, S). METEOR = stems/synonyms + order penalty. BERTScore = contextual-embedding similarity, handles paraphrase. COMET = learned from human judgments. -> BERTScore for semantic_query, not BLEU.
- LLM-as-judge: semantic, can explain, but non-deterministic and expensive: "we need a benchmark for the judge". -> validate judge against human labels (kappa); keep judge prompts fixed and cache outputs (temperature is not settable with current models).
- Takeaways: no one-size-fits-all metric; align with business goals; the metric is a proxy for the real world.

## L3 - Information Retrieval
- Entities: document, corpus, query. Inverted index maps term -> postings.
- Preprocessing must be applied identically to documents and queries.
- Sparse (TF-IDF, BM25) vs dense (embeddings). TF-IDF weighs term frequency by inverse document frequency. BM25 adds term-frequency saturation and document-length normalization; it is the default lexical workhorse. -> baseline Postgres FTS lacks IDF (W5); use real BM25.
- Scale: restrict candidates; ANN (LSH, HNSW) trades accuracy for speed. -> catalog is small enough for exact search locally.
- Bi-encoder (no interaction), cross-encoder (full interaction), late interaction (ColBERT MaxSim). Rerankers fix ordering of retrieved items.
- Matryoshka and binary embeddings: smaller vectors keep most quality. -> baseline already stores OpenAI as halfvec.
- Hybrid retrieval (sparse + dense).
- Metrics: Precision, Recall, P@k, AP, MAP, MRR (avg of 1/rank of first relevant; 0 if none), nDCG@k (DCG normalized by ideal DCG). Metrics must follow system goals. -> known-item: MRR, Success@k; candidate recall@100.

## L4 - RAG and Context Engineering
- Context engineering: decide what the model sees at each step (instructions, input, evidence, history, tools, state).
- Prompt frameworks (R.O.S.E.S., ICIO, CRISPE...) = specify task, audience, context, output contract.
- Sampling: temperature, top-k, top-p.
- Few-shot biases: majority label, recency, common token -> oversample, shuffle, paraphrase. Select prompts on dev, report on test.
- CoT and Program-of-Thought; explanations do not prove correctness.
- Prompt evaluation process: fix task/model/cases, define checks, change one factor, inspect failures, record prompt version + raw outputs + failed cases.
- RAG baseline: split, embed, store, kNN, insert into prompt. Build naive version, set up eval, iterate.
- Failures differ by stage (parse, chunk, retrieve, rank, assemble). "Find the first stage that loses required evidence before changing the prompt." -> per-stage metrics.
- Hit@k: 1 if the answer is in any of top-k.
- Document structure and metadata matter (a value needs its row/column/source). -> metadata as pre-filters.
- Chunking strategies: fixed, recursive, semantic, adaptive, model-based, overlap.
- Reranking with a cross-encoder: cannot bring back a missing candidate.
- Query methods: rewrite, multi-query (+ Reciprocal Rank Fusion, Cormack 2009), decomposition (sub-questions with dependencies), HyDE (embed a hypothetical document). -> similar_to decomposition; HyDE nice-to-have.
- Routing between sources: rules, classifier (e.g. LogisticRegression/BERT), or LLM; test paraphrases, negation, mixed intents.
- Access-controlled retrieval: enforce permissions in the application, not the prompt.
- RAG eval: measure retrieval and generation separately. Context precision/recall, contextual relevancy; answer relevancy, faithfulness (supported claims / total claims). Tools: RAGAS, DeepEval. Use controlled examples before trusting an LLM judge.
- Long tasks: persist WORKING_STATE.md (goal, constraint, evidence, open question, next step); compress stale context.
- Agentic data search: choose source + query -> search -> inspect evidence -> enough? else retry -> answer with sources.
- Indirect prompt injection via retrieved sources.

## L5 - LLMs, Safety and Ethics
- Bias types: pre-existing, technical (e.g. tokenization makes some languages worse), emergent (feedback loops, drift). -> EN vs UK gap; popularity feedback loop.
- Counterfactual evaluation: change one attribute, compare outputs. -> same query EN vs UK.
- Fairness metrics: demographic parity, equalized odds, predictive parity; impossibility theorem, but "fair enough" with tolerance.
- Signals: asymmetric quality across groups, differential refusal rates.
- Explainability: SHAP, LIME.
- Privacy: differential privacy, membership inference. -> do not send personal watch history to free third-party models.
- Alignment: SFT, RLHF, DPO. Guardrails on input/output.
- Jailbreaks and representation engineering (refusal directions, abliteration) - background only.
