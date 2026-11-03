# Evidence audit before the conference presentation

This audit checks the supplied source, scripts and saved artifacts. It does not rerun the historical experiments and does not silently rewrite the manuscript or raw results.

## Values supported by the saved Phase 3 Qwen summary

`model_results/result_v3_qwen_unrestricted.json` records 500 queries, 495 valid answers, 5 errors (1.0%), BERTScore F1 0.8460 on valid answers, corpus BLEU 14.0192, corpus chrF 31.5366, Token F1 28.96%, Recall@5 99.6%, MRR 0.996 and mean latency 18.3985 seconds. Exact match is 0.2%, not zero.

The Gemma summary records only 6 valid answers out of 500. Its 0.8792 BERTScore is computed on those successes and cannot be presented as a like-for-like quality win over Qwen's 495 valid answers. A restricted prompt is correlated with that run's formatting failures; a controlled causal explanation of internal model reasoning is not established by saved outputs.

## Manuscript conflicts and limits

1. **Retrieval:** `paper/conference_final.tex` claims 100% Recall@5 and MRR 1.000. The Qwen result file records 99.6% and 0.996. Use the artifact values unless the run is independently rechecked. More fundamentally, evaluation checks the returned **file path**, not whether the correct answer span was retrieved. With only the gold document indexed, high document-level recall is expected and does not prove clause retrieval quality.
2. **Hardware:** manuscript sections disagree on 8 GB versus 24 GB RAM and CPU-only versus GTX 1650-assisted inference. Latency is hardware-dependent. The later handover note states Windows, Ryzen 5 5600H, 8 GB RAM, GTX 1650 4 GB and Q4_K_M. Use that as documented context, while noting the saved run metadata does not independently capture hardware or quantization.
3. **Security categories:** `result_v4_qwen_security_unrestricted.json` has 50 entries, all labeled `General_Harmful`. This does not support the manuscript claim of twelve tested categories. The current `teleai_attacks.json` does not match the saved run's IDs/schema content exactly. Dataset provenance and version must be resolved.
4. **Security protocol:** the historical script sends harmful requests as user queries. It does not construct malicious retrieved documents, and it does not deterministically load a documented test corpus first. It therefore does not establish indirect prompt-injection resistance. All saved Qwen entries are judge-labeled `Refusal`; judge errors and output interpretation also need independent review. Report “0/50 saved responses labeled compliance by the project judge,” with these limits, instead of a general security guarantee.
5. **Generated attacks:** `gen_teleai.py` randomly combines locally written prompts with categories. This is synthetic data generation, not a download of the published TeleAI benchmark. Do not conflate it with a verified upstream evaluation.
6. **Faithfulness:** the saved 29.2% judge faithfulness is from a small earlier baseline run, not automatically the 500-query Qwen run. Paraphrasing can affect overlap metrics, but does not prove that unfaithful scores are false positives. BERTScore measures similarity; it does not establish factual correctness or absence of hallucinations. The historical 0.76% hallucination result belongs to a different earlier setup.
7. **Embedding comparison:** 10.97 ms versus 137.79 ms is about 12.6×, not 14×; 0.6456 versus 0.6918 differs by 0.0462, not “within 2%.” Half the vector dimension means half the raw vector storage for equal dtype and counts, not necessarily half of total application memory. MiniLM's 256-token encoding limit can truncate 350-word passages.
8. **Complexity:** brute-force cosine similarity scales linearly with indexed chunks for a fixed embedding dimension, and full sorting adds ranking cost. It is not constant-time or logarithmic retrieval.
9. **Explainability:** surfacing retrieved passages provides inspectable evidence. The current system does not verify that every generated claim is supported, nor that a model's cited source number is correct. PDF metadata identifies a chunk's starting page, not necessarily every page spanned by that chunk.
10. **Runtime change:** this desktop release runs a pinned Q4_K_M GGUF through llama-cpp-python on CPU. The original experiments used LM Studio, and the exact quantization/build is not captured in the result summary. Rerun before transferring measured performance claims to the desktop release.

## What the presentation can safely emphasize

The implemented pipeline, local execution after setup, explicit source inspection, experimental comparison process, and Qwen's observed formatting reliability are useful concrete contributions. Present numerical results with the original protocol and caveats. Distinguish measured observations from explanations or future work. Human usability, comprehension improvement, expert factuality review, multi-document retrieval and stronger adversarial evaluation remain work to do.

The paper references six figure image files that are absent from the supplied workspace. Figure planning notes are included, but an independently compilable paper package needs those assets.
