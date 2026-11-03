# Conference presentation starting point

This is a slide outline for the later PPT, not a finished presentation. Confirm conference duration, presenter name and final manuscript before generating the deck.

1. **Title and motivation:** CivicSense AI; legal/civic documents are difficult for non-expert readers. Use the manuscript's confirmed authors and affiliation.
2. **Problem and objective:** make one uploaded document easier to inspect through plain-language answers and visible source passages.
3. **Architecture:** PDF/DOCX/TXT → extraction → 350-word / 50-overlap chunks → MiniLM 384-dimensional embeddings → cosine Top-5 retrieval → local Qwen → answer, explanation, source passages.
4. **Engineering choices:** local execution and modest model size; explain the embedding and chunking comparisons without unsupported superlatives.
5. **Model selection:** baseline, restricted prompt, then larger evaluation. Highlight Qwen's 495/500 valid answers; Gemma's quality score covers only 6 valid answers.
6. **Evaluation protocol:** four LegalBench-RAG subsets, 125 queries each, one ground-truth document indexed at a time. Explain that saved Recall@5 checks document paths.
7. **Observed results:** 1.0% operational errors; 0.8460 valid-answer BERTScore; 14.0192 corpus BLEU; 18.3985-second historical mean latency; 99.6% document-path Recall@5. Do not call formatting errors hallucinations.
8. **Desktop demonstration:** launch; model preparation screen; upload the fictional sample policy; ask cancellation notice; inspect evidence; ask an unanswered parking question.
9. **Robustness and limits:** 0/50 saved responses labeled compliance by the judge; provenance/category issues and no verified indirect injection test. No universal safety guarantee.
10. **Limitations and next steps:** factuality and user studies, correct answer-span retrieval, hardware/quantization replication, adversarial documents, native release validation.
11. **Contribution and repository:** a local document-understanding pipeline and inspectable engineering record; link to https://github.com/Hari-Sri-T/CivicSense-Desktop.
12. **Questions:** keep architecture and model comparison as backup slides.

Before finalizing slides, resolve the discrepancies in EVIDENCE_AUDIT.md and confirm the original hardware. Do not use screenshots or latency from fake test responses as model evaluation evidence.
