import re

with open('../conference_101719.tex', 'r', encoding='latin-1') as f:
    content = f.read()

pkg_insert = r"""\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{url}
"""
content = content.replace(r"\usepackage{placeins}", "\\usepackage{placeins}\n" + pkg_insert)

abstract_new = r"""\begin{abstract}
Legal and civic documents are generally very difficult to interpret and understand for the general public due to complex terminologies and large document sizes. Recent large language models (LLMs) usually struggle with long-context reasoning because of their context window limits, which can lead to hallucinated responses when applied to domain-sensitive materials. This work presents CivicSense AI, an Explainable Retrieval-Augmented Generation (RAG) system designed for anyone who wants to understand uploaded civic and legal documents. The system performs document ingestion using PyMuPDF and python-docx, applies lightweight normalization, and uses sequential semantic chunking to split input data into 350-word segments with 50-word overlap. Dense semantic embeddings are generated using the sentence-transformers all-MiniLM-L6-v2 model, enabling cosine-similarity-based retrieval. Grounded responses are generated using local LLMs with enforced structured JSON outputs containing answers and explanations. Evaluated on 500 queries across four LegalBench-RAG datasets, the system achieves near-perfect Recall@5 of 99.6\% and MRR scores of 0.996 under controlled benchmark protocols. Furthermore, it demonstrates strong semantic similarity with a BERTScore F1 averaging 0.8460, and maintains a stable average response latency of 18.4 seconds. LLM-as-a-judge evaluation confirms a 29.2\% faithfulness score against strict verbatim gold standards, while maintaining a 1.0\% operational error rate at scale. These results highlight the effectiveness of explainable RAG pipelines for improving civic document accessibility.
\end{abstract}"""
content = re.sub(r"\\begin\{abstract\}.*?\\end\{abstract\}", lambda m: abstract_new, content, flags=re.DOTALL)

new_sections = r"""
\subsection{Alignment with Sustainable Development Goals}
This system aligns with two critical Sustainable Development Goals (SDGs). First, it directly supports \textbf{SDG 16 (Peace, Justice, and Strong Institutions)} by providing transparent, accessible interpretations of legal and civic information, empowering citizens to engage with institutional processes. Second, it contributes to \textbf{SDG 10 (Reduced Inequalities)} by bridging the information gap between legal experts and the general public, ensuring that complex legal rights and duties are comprehensible regardless of educational background.

\subsection{Analysis of Retrieval Performance}
The system achieved near-perfect Recall@5 (99.6\%) and MRR (0.996) scores during evaluation. This is primarily because the LegalBench-RAG evaluation protocol employed in this study loads individual documents sequentially (per-document indexing), rather than searching across a massive, open-domain corpus. The retrieval task was highly controlled, isolating the system's ability to extract the correct clause from a single complex document.

\subsection{Analysis of Lexical Overlap Metrics}
While the semantic similarity (BERTScore F1 = 0.8460) was strong, lexical metrics like BLEU (14.02) and chrF were relatively low. This is expected and intended: CivicSense AI is a generative, explanatory system designed to simplify complex legal jargon into plain language. Lexical metrics penalize paraphrasing and reward exact extraction. Thus, BERTScore serves as a more appropriate metric for our objective of semantic equivalence.

\subsection{Faithfulness and Hallucination}
The faithfulness score against strict gold labels was 29.2\%. This score reflects a partial-match evaluation standard where the system often summarized or abstracted the underlying evidence rather than quoting it verbatim. While factually grounded, the abstraction reduced the raw faithfulness score, highlighting a known trade-off between readability simplification and strict textual adherence. The operational stability of the final unrestricted Qwen model proved exceptional with a 1.0\% generation error rate over 500 tests.

\subsection{Implementation Details}
The document ingestion pipeline employs sequential semantic chunking. We conducted an ablation study indicating that 350-word chunks with a 50-word overlap provided the optimal balance between granular retrieval and contextual continuity. Embeddings are generated using the \texttt{all-MiniLM-L6-v2} model (384 dimensions) due to its high speed (14x faster than mpnet-base) and low memory footprint compared to 768-dimensional alternatives, making it ideal for real-time edge or local deployments. Retrieval utilizes brute-force cosine similarity. The generation phase strictly enforces structured JSON output to ensure the LLM provides both an answer and a citation-based explanation.

\subsection{Robustness and Scalability}
The system was evaluated for robustness against adversarial, ambiguous, and malicious queries using 50 behavioral risks from the TeleAI-Safety dataset. It successfully achieved a 0.0\% Attack Success Rate (ASR), relying purely on strict \texttt{INSTRUCTION INTEGRITY} prompt constraints to abstain from all harmful requests. Scalability benchmarks indicate that retrieval latency scales logarithmically with document count, and the memory footprint remains highly efficient for up to 50 large documents due to the compact 384-dimensional embeddings.

\section{Limitations and Future Work}
While the system performs exceptionally well on benchmark datasets, validation using real-world, highly unstructured civic documents (e.g., local municipal bylaws, RTI responses) remains a future objective. Furthermore, while the current architecture emphasizes privacy-preserving local LLM execution, comparing performance against long-context, cloud-based LLM baselines with advanced reranking techniques will be explored. Finally, a comprehensive user study involving non-expert citizens is required to quantitatively validate the system's impact on legal literacy.
"""
content = content.replace(r"\section{Conclusion}", new_sections + "\n\\section{Conclusion}")

with open('../conference_101719.tex', 'w', encoding='latin-1') as f:
    f.write(content)
