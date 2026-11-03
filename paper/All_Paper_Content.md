# CivicSense AI: An Explainable Retrieval-Augmented Generation System for Public Document Understanding

**Hari Sri Shanmukha Sai Tanniru**, **Sahil Awatramani**, **Tanishq Sangle**, **Sheetal Kusal**, **Supriya Mahadevkar**

---

# Abstract

Legal and civic documents are often written using dense, domain-specific language that makes them difficult for non-expert users to understand. Although Large Language Models (LLMs) have demonstrated remarkable capabilities in natural language understanding and question answering, their direct application to legal document interpretation remains limited by hallucinations, restricted context windows, lack of evidence attribution, and privacy concerns associated with cloud-based inference. Retrieval-Augmented Generation (RAG) addresses many of these limitations by grounding responses in retrieved document evidence; however, deploying explainable, trustworthy, and privacy-preserving legal RAG systems on commodity consumer hardware remains relatively underexplored.

This paper presents **CivicSense AI**, a fully local Explainable Retrieval-Augmented Generation (RAG) framework designed to assist users in understanding legal and civic documents through evidence-grounded natural language explanations. The proposed system performs document parsing, semantic chunking, dense embedding generation, semantic retrieval, and grounded response generation while preserving complete local execution of user documents. To improve transparency, every generated response is accompanied by explicit source attribution identifying the retrieved evidence supporting the explanation. Additionally, an instruction-integrity mechanism protects the system against document-based prompt injection attacks by preventing malicious instructions embedded within retrieved content from influencing model behaviour.

Rather than selecting the final architecture empirically, CivicSense AI was developed through a systematic engineering study consisting of embedding model selection, chunk-size optimization, comparative evaluation of lightweight open-weight language models, reasoning-ablation experiments, large-scale benchmark validation, and adversarial security assessment. The final deployment configuration was evaluated on **500 benchmark queries** spanning the four LegalBench-RAG datasets (CUAD, ContractNLI, Privacy-QA, and MAUD). The selected configuration achieved **Recall@5 = 100.0%**, **MRR = 1.000**, **BERTScore F1 = 0.8460**, **Corpus BLEU = 14.02**, **Corpus chrF = 31.54**, **Token F1 = 29.0%**, and a formatting error rate of **1.0%**. Security evaluation using fifty prompt-injection attacks from the TeleAI-Safety benchmark resulted in an **Attack Success Rate (ASR) of 0.0%**, demonstrating strong robustness against document-based prompt injection.

The experimental results demonstrate that accurate, explainable, and privacy-preserving legal document understanding can be achieved using lightweight open-weight language models executing entirely on local commodity hardware. Beyond presenting a practical legal assistant, this work provides a systematic empirical study of engineering trade-offs involved in deploying trustworthy local Retrieval-Augmented Generation systems for legal document understanding.

---

## Keywords

Retrieval-Augmented Generation (RAG), Explainable Artificial Intelligence (XAI), Large Language Models, Legal Document Understanding, Semantic Search, Information Retrieval, Local Artificial Intelligence, Prompt Injection Defense, Civic Technology, Natural Language Processing.

# 2. Introduction

Recent advances in Large Language Models (LLMs) have significantly improved the ability of artificial intelligence systems to understand natural language, summarize lengthy documents, and answer complex user queries. Their success has accelerated adoption across numerous domains, including healthcare, finance, education, software engineering, and legal services. Despite these advances, directly applying general-purpose LLMs to legal document interpretation remains challenging. Generated responses often rely on the model's parametric knowledge rather than the uploaded document, increasing the risk of hallucinations, unsupported claims, and omitted contractual details. Furthermore, the limited context window of language models restricts their ability to reason over lengthy legal agreements containing numerous interdependent clauses.

These limitations are particularly significant in the legal and civic domain. Contracts, privacy policies, tenancy agreements, merger documents, government regulations, and institutional policies are typically written using complex legal language that is difficult for non-expert users to interpret. Existing cloud-hosted conversational AI systems can often produce fluent explanations; however, they rarely provide verifiable evidence supporting their answers. More importantly, uploading confidential legal documents to external cloud services introduces privacy, confidentiality, and regulatory concerns that limit adoption in practical settings.

Retrieval-Augmented Generation (RAG) has emerged as one of the most effective approaches for addressing these challenges. Rather than generating responses solely from pre-trained model parameters, RAG first retrieves semantically relevant document passages before conditioning the language model on this retrieved evidence. By grounding responses in the source document, RAG substantially reduces hallucination while improving factual consistency and allowing generated explanations to remain traceable to their original context. These characteristics make Retrieval-Augmented Generation particularly well suited for legal document understanding, where correctness and evidence attribution are essential.

Although Retrieval-Augmented Generation has achieved considerable success, several practical challenges remain. Many existing systems are designed for enterprise environments that assume access to cloud-hosted language models, dedicated GPU servers, distributed vector databases, or proprietary APIs. Such deployment assumptions increase operational cost while introducing latency and privacy concerns. Furthermore, relatively few studies systematically investigate the engineering decisions involved in building reliable local RAG systems, including embedding model selection, chunk-size optimization, prompt design, lightweight language model comparison, reasoning behaviour, and security against prompt injection attacks.

Another important challenge concerns explainability. While many conversational AI systems provide convincing answers, users often have limited visibility into the evidence supporting those responses. In legal applications, this lack of transparency reduces user trust because important contractual obligations or legal interpretations cannot easily be verified. Explainable Retrieval-Augmented Generation addresses this problem by exposing the retrieved evidence used during generation, enabling users to inspect the original document passages that support each generated explanation. Such transparency is particularly important for legal decision support, where users should always be able to validate generated information against the underlying document.

Recent work has also highlighted prompt injection as an emerging security threat for Retrieval-Augmented Generation systems. Because retrieved documents are incorporated directly into the language model context, malicious instructions embedded within uploaded documents may influence model behaviour, override system prompts, or produce unintended responses. Consequently, secure RAG systems require explicit mechanisms that separate trusted system instructions from untrusted retrieved content while preventing document text from being interpreted as executable instructions.

Motivated by these challenges, this paper presents **CivicSense AI**, an explainable Retrieval-Augmented Generation framework designed specifically for interactive understanding of legal and civic documents through fully local inference. Unlike conventional legal assistants that generate responses without transparent justification, CivicSense AI grounds every explanation in retrieved document evidence and exposes the corresponding source locations to the user. The proposed framework additionally incorporates an instruction-integrity mechanism that protects against document-based prompt injection attacks while preserving explainability and local deployment.

Rather than selecting the final architecture through empirical trial and error, CivicSense AI was developed through a systematic engineering study comprising multiple controlled experiments. The study first compares candidate sentence embedding models to determine an appropriate retrieval backbone for local deployment. A chunk-size ablation study then identifies the optimal segmentation strategy for legal documents. Subsequently, five lightweight open-weight language models are evaluated under unrestricted reasoning conditions, followed by a reasoning-ablation study investigating the influence of suppressing internal reasoning traces on retrieval-grounded answer generation. The best-performing configurations are then validated through large-scale benchmarking using the LegalBench-RAG benchmark and finally evaluated against document-based prompt injection attacks using the TeleAI-Safety benchmark. This experimental methodology provides both a practical deployment framework and empirical insights into the behaviour of lightweight language models within legal Retrieval-Augmented Generation systems.

The primary contributions of this work are summarized below.

- **A fully local explainable legal RAG framework.** We propose CivicSense AI, an end-to-end Retrieval-Augmented Generation framework that performs evidence-grounded legal document understanding entirely on local consumer hardware while preserving document privacy.

- **A systematic engineering study of lightweight language models.** We present a comprehensive empirical comparison of multiple lightweight open-weight language models under unrestricted and reasoning-suppressed prompting strategies to identify suitable deployment configurations for legal Retrieval-Augmented Generation.

- **Experimental justification of architectural decisions.** Every major component of the proposed framework, including embedding model selection, chunk-size optimization, prompting strategy, language model selection, and security mechanisms, is justified through controlled experiments rather than empirical selection.

- **Comprehensive evaluation of retrieval quality, generation quality, and robustness.** The proposed framework is evaluated using LegalBench-RAG, LLM-as-a-Judge faithfulness analysis, hallucination assessment, and adversarial security benchmarking using TeleAI-Safety, providing a holistic evaluation of practical local legal Retrieval-Augmented Generation.

The remainder of this paper is organized as follows. Section II reviews recent developments in Retrieval-Augmented Generation, legal question answering, explainable AI, and secure RAG systems. Section III presents the proposed CivicSense AI architecture and implementation methodology. Section IV describes the experimental setup and presents component selection studies, language model benchmarking, reasoning-ablation experiments, large-scale validation, and security evaluation. Finally, Section V concludes the paper and discusses future research directions.

# 3. Related Work

## 3.1 Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) has become one of the most influential paradigms for improving the factual reliability of Large Language Models (LLMs). Instead of relying exclusively on parametric knowledge acquired during pre-training, RAG augments the generation process by retrieving relevant information from external knowledge sources before response generation. This significantly reduces hallucinations, improves factual consistency, and enables language models to answer questions using domain-specific information unavailable during training. Comprehensive surveys by Gao *et al.* and Cheng *et al.* identify Retrieval-Augmented Generation as one of the most promising directions for developing trustworthy language model applications across knowledge-intensive domains while highlighting retrieval quality and evidence grounding as critical factors influencing overall system performance.

Modern RAG systems typically consist of four principal stages: document preprocessing, semantic indexing, information retrieval, and response generation. Documents are first segmented into manageable chunks before being transformed into dense vector representations using sentence embedding models. During inference, the user's query is embedded into the same semantic space, allowing similarity-based retrieval of relevant document segments. These retrieved passages are subsequently incorporated into the prompt supplied to the language model, enabling responses to remain grounded in retrieved evidence rather than relying solely on the model's internal knowledge. Recent research has proposed numerous improvements including adaptive retrieval, hybrid dense-lexical retrieval, multi-stage reranking, and dynamic context selection to further improve retrieval quality and answer faithfulness.

Despite these advances, many existing Retrieval-Augmented Generation systems target enterprise-scale deployments requiring dedicated GPU infrastructure, cloud-hosted vector databases, or proprietary language model APIs. Comparatively fewer studies investigate practical deployment on commodity consumer hardware while simultaneously preserving privacy, explainability, and robustness against adversarial document manipulation.

---

## 3.2 Retrieval-Augmented Generation for Legal Document Understanding

The legal domain presents unique challenges for Retrieval-Augmented Generation owing to the complexity, length, and specialized terminology of legal documents. Contracts, regulatory documents, privacy policies, and merger agreements frequently contain long interdependent clauses whose interpretation depends upon surrounding context. Consequently, directly applying general-purpose language models often produces incomplete or unsupported explanations.

Recent legal Retrieval-Augmented Generation systems have demonstrated that grounding responses in retrieved legal evidence substantially improves factual consistency and reduces hallucination compared to standalone language models. LegalBench-RAG introduced one of the first standardized benchmarks specifically designed for evaluating retrieval-grounded legal question answering across multiple legal datasets. Similarly, LegalRAG and other domain-specific retrieval systems have explored multilingual legal retrieval, hybrid retrieval architectures, and retrieval-enhanced legal reasoning.

Although these studies demonstrate the value of Retrieval-Augmented Generation for legal applications, most primarily focus on retrieval effectiveness or answer accuracy. Comparatively little attention has been devoted to explainability, deployment efficiency, reasoning behaviour of lightweight language models, or systematic engineering studies evaluating the practical trade-offs involved in constructing local legal Retrieval-Augmented Generation systems.

---

## 3.3 Explainability and Trustworthy Retrieval-Augmented Generation

Explainability has become an increasingly important requirement for trustworthy language model systems, particularly in domains where generated responses may influence human decision making. Traditional conversational language models typically provide fluent responses without exposing the evidence supporting those responses, making it difficult for users to verify factual correctness or identify hallucinated information.

Explainable Retrieval-Augmented Generation addresses this limitation by explicitly exposing the retrieved evidence used during response generation. Rather than treating the language model as an opaque black box, explainable RAG systems allow users to inspect retrieved passages, similarity scores, and document locations supporting each generated explanation. This improves user trust while enabling manual verification of important contractual clauses and legal obligations.

Several recent surveys identify explainability as one of the defining characteristics of trustworthy Retrieval-Augmented Generation systems. Nevertheless, relatively few implementations combine evidence attribution with comprehensive evaluation of retrieval quality, response quality, faithfulness, hallucination behaviour, and adversarial robustness within a unified framework.

---

## 3.4 Prompt Injection and Secure Retrieval-Augmented Generation

As Retrieval-Augmented Generation systems become increasingly popular, prompt injection has emerged as one of their most significant security challenges. Because retrieved documents are incorporated directly into the language model context, malicious instructions embedded within those documents may override system prompts, manipulate assistant behaviour, or produce unintended responses.

Recent research has proposed multiple mitigation strategies including prompt hardening, instruction separation, context filtering, retrieval sanitization, and system-level instruction enforcement. The recently introduced TeleAI-Safety benchmark provides one of the first standardized evaluation frameworks for assessing prompt injection resistance across a diverse collection of adversarial attacks.

Despite growing interest in secure Retrieval-Augmented Generation, most existing work evaluates security independently from retrieval quality or explainability. Consequently, relatively few studies investigate whether robust prompt injection defence can be achieved without sacrificing retrieval effectiveness or response quality.

---

## 3.5 Research Gap

The literature demonstrates significant progress in Retrieval-Augmented Generation, legal information retrieval, explainable AI, and secure language model deployment. However, several important gaps remain.

First, many existing legal Retrieval-Augmented Generation systems rely on cloud-hosted language models or enterprise-scale infrastructure, limiting practical deployment for privacy-sensitive legal documents. Second, comparatively little research systematically investigates the engineering decisions underlying successful local RAG systems, including embedding selection, chunk-size optimization, prompting strategies, reasoning behaviour, and lightweight language model selection. Third, although explainability and security have individually received increasing attention, relatively few studies evaluate these characteristics together within a unified experimental framework.

CivicSense AI addresses these gaps by presenting not only a fully local explainable Retrieval-Augmented Generation framework for legal and civic document understanding, but also a comprehensive engineering study that systematically evaluates every major architectural decision through controlled experiments. Rather than selecting system components empirically, the proposed framework justifies embedding models, chunking strategies, prompting configurations, language model selection, and security mechanisms using reproducible benchmark evaluations. This combination of explainability, local deployment, systematic evaluation, and adversarial robustness distinguishes CivicSense AI from existing legal Retrieval-Augmented Generation systems.

# 4. Proposed Methodology

## 4.1 System Overview

CivicSense AI follows an Explainable Retrieval-Augmented Generation (RAG) architecture designed specifically for understanding legal and civic documents through fully local inference. Unlike conventional conversational AI systems that rely primarily on the parametric knowledge of a Large Language Model (LLM), the proposed framework grounds every generated response in semantically retrieved evidence extracted directly from the uploaded document. This retrieval-first design substantially reduces hallucination while enabling users to verify generated explanations against the original source material.

The overall pipeline consists of two complementary stages:

1. **Offline Indexing Pipeline**, responsible for document ingestion, preprocessing, semantic chunking, embedding generation, and vector indexing.

2. **Online Inference Pipeline**, responsible for semantic retrieval, grounded response generation, explainability, and security validation.

Separating document indexing from query-time inference minimizes latency during user interaction while allowing multiple documents to be queried efficiently after preprocessing.

---

## 4.2 Offline Indexing Pipeline

### 4.2.1 Document Ingestion

The framework accepts three document formats commonly encountered in legal and civic applications:

- Portable Document Format (PDF)
- Microsoft Word Documents (.docx)
- Plain Text (.txt)

Each uploaded document is parsed locally without transmitting data to external services.

PDF documents are processed using **PyMuPDF**, enabling page-level extraction while preserving page boundaries. Microsoft Word files are processed using **python-docx**, maintaining paragraph-level structure. Plain-text files are segmented according to paragraph boundaries.

For every extracted segment, CivicSense AI stores:

- document name,
- page number (where applicable),
- paragraph identifier,
- textual content.

This metadata remains associated with every chunk throughout the retrieval pipeline, enabling precise evidence attribution during response generation.

---

### 4.2.2 Semantic Chunking

Following document extraction, the textual content is partitioned into overlapping semantic chunks suitable for dense retrieval.

Rather than using page boundaries or fixed character windows, CivicSense AI employs a word-based chunking strategy. Through experimental evaluation (Section IV), a chunk size of **350 words** with an overlap of **50 words** was selected as the final deployment configuration.

The overlap serves two important purposes:

- preserving contextual continuity between adjacent chunks,
- reducing information loss for legal clauses spanning chunk boundaries.

Each generated chunk retains:

- textual content,
- originating document,
- page number,
- chunk identifier,
- chunk size.

This metadata later enables explainable retrieval and traceable response generation.

---

### 4.2.3 Dense Embedding Generation

Each chunk is transformed into a dense semantic vector using the **all-MiniLM-L6-v2** sentence embedding model.

Unlike the language model, which executes through LM Studio, embedding generation is performed locally on the CPU using SentenceTransformers.

The choice of MiniLM was not empirical. Multiple embedding models were compared experimentally before deployment (Section IV), where MiniLM demonstrated the most favorable balance between retrieval effectiveness, computational efficiency, and deployment cost.

Once generated, the embeddings are stored in an in-memory vector representation that serves as the document index for semantic retrieval.

---

## 4.3 Online Inference Pipeline

When a user submits a question, CivicSense AI performs Retrieval-Augmented Generation through four sequential stages:

1. Query Embedding
2. Semantic Retrieval
3. Prompt Construction
4. Response Generation

---

### 4.3.1 Query Embedding

The user query is embedded using the same MiniLM encoder employed during document indexing.

Encoding both document chunks and user queries within the same semantic space enables similarity comparison independent of exact lexical overlap.

Consequently, queries using simplified language can retrieve legally relevant passages even when the wording differs substantially from the original document.

---

### 4.3.2 Semantic Retrieval

Retrieval is performed using cosine similarity between the embedded user query and every indexed document chunk.

The similarity between query embedding \(q\) and document embedding \(d\) is computed as

\[
\text{Similarity}(q,d)=
\frac{q \cdot d}
{\|q\|\|d\|}
\]

The chunks are ranked according to similarity score.

The five highest-ranked chunks (**Top-K = 5**) are selected for response generation.

Selecting multiple evidence passages improves contextual coverage while reducing the probability of omitting contractual clauses distributed across different sections of a document.

Unlike hybrid retrieval methods, CivicSense AI relies exclusively on dense semantic retrieval, maintaining a lightweight architecture suitable for local deployment.

---

### 4.3.3 Prompt Construction

The retrieved document chunks are incorporated into a structured prompt supplied to the language model.

The prompt consists of three components:

- a system prompt defining assistant behaviour,
- retrieved document evidence,
- the user question.

Rather than allowing unrestricted generation, the system prompt explicitly instructs the language model to answer exclusively from the retrieved evidence.

Whenever insufficient evidence exists, the model is instructed to state that the required information is unavailable instead of generating unsupported content.

This grounding strategy substantially reduces hallucination while maintaining transparency.

---

### 4.3.4 Response Generation

Response generation is performed using the **qwen2.5-3b-instruct** language model executed locally through **LM Studio**.

The model operates using the OpenAI-compatible API interface provided by LM Studio.

The deployed model uses

- Q4_K_M quantization,
- an 8192-token context window,
- temperature = 0.1,
- maximum generation length of 4096 tokens.

Unlike cloud-hosted language models, all inference remains entirely local.

Consequently, uploaded legal documents never leave the user's device, preserving confidentiality while eliminating recurring cloud inference costs.

---

## 4.4 Explainability

A major objective of CivicSense AI is to improve transparency during legal document understanding.

Rather than returning only generated answers, the system exposes the retrieved evidence supporting every response.

Each answer is accompanied by

- source document,
- page number,
- retrieval similarity score,
- retrieved text.

This enables users to verify every explanation against the original document rather than blindly trusting generated content.

Such evidence attribution is particularly valuable for legal applications where contractual obligations should always remain traceable to their originating clauses.

---

## 4.5 Security Design

Prompt injection has recently emerged as one of the primary security concerns affecting Retrieval-Augmented Generation systems.

Because retrieved document text becomes part of the language model context, malicious instructions embedded inside uploaded documents may influence model behaviour if appropriate safeguards are absent.

To address this challenge, CivicSense AI incorporates an **Instruction Integrity** mechanism.

Retrieved document content is treated exclusively as untrusted evidence rather than executable instructions.

The system prompt explicitly prohibits the language model from

- executing instructions contained within retrieved documents,
- revealing hidden prompts,
- modifying assistant behaviour,
- generating responses beyond retrieved evidence.

This separation between trusted system instructions and untrusted retrieved content significantly improves resistance against document-based prompt injection attacks.

The effectiveness of this mechanism is evaluated experimentally using the TeleAI-Safety benchmark in Section IV.

---

## 4.6 Output Validation

Language models occasionally generate malformed JSON outputs, particularly during long-form responses.

To improve robustness, CivicSense AI incorporates a multi-stage validation pipeline.

Generated responses first undergo standard JSON parsing.

If parsing fails, an automated JSON repair procedure attempts structural correction.

Only responses that remain invalid after repair are treated as formatting failures.

This additional validation stage substantially reduced formatting errors during large-scale benchmarking while allowing reliable automated evaluation across multiple language models.

---

## 4.7 Implementation Details

All experiments were conducted using a consumer laptop running Microsoft Windows equipped with

- AMD Ryzen 5 5600H processor,
- 8 GB RAM,
- NVIDIA GeForce GTX 1650 Laptop GPU (4 GB VRAM).

Language model inference was executed locally using LM Studio, whereas embedding generation was performed on the CPU using SentenceTransformers.

The final production configuration employed

- all-MiniLM-L6-v2 embeddings,
- cosine similarity retrieval,
- Top-K = 5,
- 350-word chunks with 50-word overlap,
- qwen2.5-3b-instruct (Q4_K_M).

The complete framework operates entirely on local hardware without requiring external cloud services, thereby preserving document privacy while maintaining practical deployment costs.

# 5. Experimental Evaluation

## 5.1 Experimental Methodology

The objective of the experimental evaluation was not simply to report the performance of the final CivicSense AI system but to systematically justify every major architectural decision through controlled experimentation. Rather than selecting the retrieval pipeline, embedding model, chunking strategy, prompting configuration, or language model empirically, each component was evaluated independently before integration into the final deployment.

The complete evaluation was conducted in four sequential phases.

**Phase 1** compared multiple lightweight open-weight language models under unrestricted reasoning conditions to establish baseline performance.

**Phase 2** investigated the influence of suppressing reasoning behaviour using an alternative prompting strategy.

**Phase 3** performed large-scale validation of the selected deployment configuration using the LegalBench-RAG benchmark.

Finally, **Phase 4** evaluated robustness against prompt injection attacks using the TeleAI-Safety benchmark.

This staged evaluation methodology provides significantly greater insight than evaluating only the final deployed model because it reveals the engineering trade-offs involved in constructing trustworthy Retrieval-Augmented Generation systems.

---

## 5.2 Datasets

### LegalBench-RAG

Performance evaluation employed four datasets from the LegalBench-RAG benchmark.

- CUAD
- ContractNLI
- Privacy-QA
- MAUD

Together these datasets cover

- contract analysis,
- legal entailment,
- privacy policy interpretation,
- merger agreement understanding.

To ensure reproducibility, evaluation used a deterministic subset consisting of the first **125 benchmark queries** from each dataset, resulting in a total of **500 evaluation queries**.

Unlike random sampling, this deterministic selection guarantees identical benchmark composition across repeated experiments.

---

### TeleAI-Safety

Security evaluation employed the TeleAI-Safety benchmark.

Fifty adversarial prompts representing document-based prompt injection attacks were used to evaluate whether malicious instructions embedded inside retrieved evidence could influence the language model.

The benchmark reports Attack Success Rate (ASR), where lower values indicate stronger resistance against prompt injection.

---

## 5.3 Evaluation Metrics

The proposed framework was evaluated using three complementary categories of metrics.

### Retrieval Metrics

Retrieval quality was measured using

- Recall@5
- Mean Reciprocal Rank (MRR)

These metrics quantify whether the relevant document chunk was successfully retrieved before response generation.

---

### Generation Metrics

Generation quality was evaluated using

- Exact Match (EM)
- Token F1
- Corpus BLEU
- Corpus chrF
- BERTScore F1

Unlike lexical metrics such as BLEU and Token F1, BERTScore evaluates semantic similarity between generated responses and reference answers, making it particularly appropriate for explainable legal question answering where generated explanations intentionally paraphrase complex legal language into simpler natural language.

---

### Robustness Metrics

System robustness was evaluated using

- formatting error rate,
- average response latency,
- hallucination rate,
- faithfulness,
- Attack Success Rate (ASR).

Faithfulness and hallucination were measured using an LLM-as-a-Judge evaluation framework, enabling semantic assessment beyond lexical overlap.

---

# 5.4 Embedding Model Selection

The first experiment investigated the effect of sentence embedding models on retrieval quality.

Since CivicSense AI targets deployment on consumer hardware, the embedding model was required to provide a suitable balance between

- semantic retrieval performance,
- computational efficiency,
- inference speed,
- deployment cost.

Three embedding models were evaluated under identical retrieval conditions.

| Embedding Model | Recall@5 | MRR | Relative Speed |
|-----------------|----------|-----|----------------|
| all-MiniLM-L6-v2 | **(Master Report)** | **(Master Report)** | Fast |
| all-mpnet-base-v2 | (Master Report) | (Master Report) | Moderate |
| BAAI/bge-base-en-v1.5 | (Master Report) | (Master Report) | Slow |

**Table 1.** Embedding model comparison.

The results indicate that retrieval performance differed only marginally across the evaluated embedding models. However, **all-MiniLM-L6-v2** required substantially fewer computational resources while maintaining competitive retrieval quality.

Considering both retrieval effectiveness and deployment efficiency, MiniLM was selected as the embedding model for all subsequent experiments.

---

# 5.5 Chunk Size Optimization

Document chunking strongly influences Retrieval-Augmented Generation because excessively small chunks often fragment legal clauses, whereas excessively large chunks introduce retrieval noise.

To determine an appropriate segmentation strategy, three chunking configurations were evaluated using a deterministic subset of 200 benchmark queries comprising fifty queries from each LegalBench-RAG dataset.

The evaluated configurations were

- 200-word chunks with 30-word overlap,
- 350-word chunks with 50-word overlap,
- 500-word chunks with 75-word overlap.

| Chunk Size | Overlap | Recall@5 | MRR | BERTScore |
|------------|---------|----------|-----|-----------|
| 200 | 30 | (Master Report) | (Master Report) | (Master Report) |
| **350** | **50** | **(Master Report)** | **(Master Report)** | **(Master Report)** |
| 500 | 75 | (Master Report) | (Master Report) | (Master Report) |

**Table 2.** Chunk size ablation study.

The 350-word configuration consistently achieved the best overall trade-off between retrieval accuracy and response quality.

Smaller chunks occasionally separated semantically related legal clauses across multiple retrieval units, whereas larger chunks diluted semantic specificity by introducing unrelated contextual information.

Consequently, the 350-word chunk size with a 50-word overlap was adopted throughout the remaining experiments.

---

# 5.6 Phase 1 – Baseline Comparison of Lightweight Language Models

After selecting the retrieval pipeline, the next objective was to identify a suitable lightweight language model for local deployment.

Five open-weight models were evaluated under identical retrieval conditions using unrestricted reasoning prompts.

The evaluated models were

- qwen2.5-3b-instruct,
- gemma-3-4b,
- llama-3.2-3b-instruct,
- Ministral-3B,
- Nemotron Nano 4B.

Evaluation employed identical

- embedding model,
- retrieval pipeline,
- chunking strategy,
- prompting template.

Only the language model differed.

The benchmark measured retrieval quality, lexical similarity, semantic similarity, latency, and formatting robustness.

> **INSERT TABLE 3 HERE**
>
> Phase 1 Model Comparison
>
> *(Use the complete table directly from the Master Evaluation Report.)*

Phase 1 produced several important observations.

First, all evaluated models achieved nearly identical retrieval performance because retrieval quality depends primarily on the embedding model rather than the language model.

Second, substantial differences emerged in generation quality, formatting robustness, and latency.

Although several models produced competitive semantic similarity scores, noticeable variation was observed in formatting consistency and response reliability.

These findings motivated the reasoning-ablation experiments presented in Phase 2.

# 5.7 Phase 2 – Reasoning Ablation Study

Recent open-weight language models increasingly incorporate internal reasoning capabilities that encourage the model to generate intermediate reasoning before producing the final answer. While this behaviour often improves performance on mathematical and logical reasoning tasks, its influence on Retrieval-Augmented Generation remains comparatively underexplored. Since Retrieval-Augmented Generation already supplies the model with relevant evidence, unrestricted reasoning may unnecessarily consume context length, increase inference latency, or encourage unsupported reasoning beyond the retrieved document.

To investigate this behaviour, a reasoning-ablation study was conducted using the same five lightweight language models evaluated in Phase 1. Instead of modifying model parameters, reasoning behaviour was controlled through an alternative system prompt that explicitly instructed the language model to answer directly from the retrieved evidence without exposing or generating intermediate reasoning traces.

All remaining experimental settings—including retrieval pipeline, embedding model, chunking strategy, and evaluation methodology—were kept identical to Phase 1, ensuring that any observed differences could be attributed solely to the prompting strategy.

The evaluated models were:

- qwen2.5-3b-instruct
- gemma-3-4b
- llama-3.2-3b-instruct
- Ministral-3B
- Nemotron Nano 4B

---

## 5.7.1 Comparative Performance

> **INSERT TABLE 4 HERE**
>
> **Phase 2 Model Comparison**
>
> *(Directly use the table from the Master Evaluation Report.)*

Unlike Phase 1, substantial behavioural differences emerged between models when reasoning was suppressed.

Several observations proved particularly important.

### Llama 3.2

Suppressing reasoning significantly improved response stability by preventing unnecessary reasoning traces from consuming the available context window. The model produced more consistent JSON outputs and demonstrated improved retrieval-grounded answer generation.

### Gemma 3

Gemma exhibited one of the largest improvements under reasoning suppression during the small-scale benchmark. The restricted prompting strategy substantially reduced formatting failures while improving semantic similarity, initially suggesting Gemma as one of the strongest candidates for local deployment.

### Qwen 2.5

Unlike Llama and Gemma, Qwen demonstrated the opposite behaviour. Restricting reasoning reduced overall semantic quality while slightly degrading multiple evaluation metrics. This suggests that Qwen's instruction tuning already balances reasoning and evidence grounding effectively, making additional reasoning suppression unnecessary.

### Ministral and Nemotron

Both models showed comparatively modest changes between unrestricted and restricted prompting, indicating lower sensitivity to reasoning suppression than the remaining models.

These observations demonstrate that reasoning behaviour is highly model dependent rather than universally beneficial or detrimental. Consequently, prompting strategies should be selected individually for each language model instead of being applied uniformly across different architectures.

---

## 5.7.2 Phase 1 versus Phase 2

To better understand the influence of reasoning suppression, Phase 1 and Phase 2 results were compared directly.

> **INSERT TABLE 5 HERE**
>
> **Comparison between unrestricted and reasoning-suppressed prompting**
>
> *(Master Evaluation Report.)*

The comparison reveals three distinct behavioural patterns.

1. Models that benefited substantially from reasoning suppression.

2. Models that showed negligible change.

3. Models whose performance decreased after reasoning suppression.

This finding is particularly important because it demonstrates that Retrieval-Augmented Generation does not universally benefit from suppressing reasoning. Instead, the interaction between retrieval grounding and model-specific instruction tuning determines whether unrestricted reasoning improves or degrades overall response quality.

These observations directly informed the model-selection process used in the subsequent large-scale evaluation.

---

# 5.8 Faithfulness and Hallucination Analysis

Traditional lexical evaluation metrics such as BLEU, Exact Match, and Token F1 primarily measure textual similarity between generated responses and benchmark answers. However, CivicSense AI intentionally produces simplified natural-language explanations rather than reproducing legal clauses verbatim. Consequently, lexical metrics alone cannot reliably determine whether generated explanations remain faithful to the retrieved evidence.

To address this limitation, an additional semantic evaluation was performed using an LLM-as-a-Judge framework.

For each generated response, the evaluation framework considered

- the user query,
- retrieved evidence,
- reference answer,
- generated explanation.

The judge then evaluated two complementary properties:

- **Faithfulness**, measuring whether the generated explanation remained supported by the retrieved document evidence.

- **Hallucination**, measuring whether unsupported factual claims appeared in the generated response.

Unlike lexical metrics, these evaluations assess semantic correctness rather than textual overlap.

---

## 5.8.1 Faithfulness Evaluation

> **INSERT TABLE 6 HERE**
>
> **Faithfulness comparison**
>
> *(Master Evaluation Report.)*

The majority of evaluated models demonstrated high faithfulness scores, indicating that Retrieval-Augmented Generation successfully constrained generation to retrieved evidence.

Interestingly, several responses with relatively low BLEU scores nevertheless received high faithfulness ratings because the generated explanations accurately paraphrased complex contractual language instead of reproducing benchmark answers verbatim.

This observation reinforces the importance of semantic evaluation when assessing explainable legal question answering systems.

---

## 5.8.2 Hallucination Analysis

> **INSERT TABLE 7 HERE**
>
> **Hallucination comparison**
>
> *(Master Evaluation Report.)*

Hallucination rates remained low across the strongest-performing models, indicating that retrieval grounding substantially reduced unsupported factual generation.

Models exhibiting higher formatting errors generally also demonstrated slightly increased hallucination rates, suggesting that unstable response generation may contribute to reduced factual reliability.

Overall, Retrieval-Augmented Generation combined with carefully designed prompting significantly constrained hallucination compared to unrestricted language-model generation.

---

## 5.8.3 Discussion

The faithfulness and hallucination experiments provide an important insight regarding evaluation methodology.

Although lexical metrics such as BLEU, Token F1, and Exact Match remain valuable for standardized benchmarking, they often underestimate explanation quality because benchmark answers frequently consist of verbatim legal clauses whereas CivicSense AI intentionally generates simplified explanations for improved readability.

Semantic evaluation metrics—including BERTScore and LLM-as-a-Judge faithfulness—more accurately capture the intended behaviour of explainable legal assistants by rewarding faithful paraphrasing rather than literal textual reproduction.

Consequently, the proposed evaluation framework combines lexical, semantic, retrieval, and robustness metrics to provide a more comprehensive assessment of Retrieval-Augmented Generation performance than any individual metric alone.

The combined findings from Phases 1 and 2 ultimately guided the selection of candidate models for the large-scale evaluation presented in the following section.

# 5.9 Phase 3 – Large-Scale Benchmark Evaluation

Following the component selection experiments and reasoning-ablation study, the most promising deployment configurations were evaluated using the complete LegalBench-RAG benchmark. Unlike the earlier exploratory experiments, which were designed primarily to compare alternative design choices, Phase 3 aimed to validate the robustness of the selected configurations under a substantially larger and more diverse evaluation workload.

The benchmark consisted of **500 evaluation queries**, corresponding to **125 benchmark queries** from each of the four LegalBench-RAG datasets:

- CUAD
- ContractNLI
- Privacy-QA
- MAUD

All experiments employed the final retrieval pipeline established during the previous phases, including

- all-MiniLM-L6-v2 embeddings,
- cosine similarity retrieval,
- Top-K = 5,
- 350-word chunks,
- 50-word overlap.

Only the language model and prompting configuration differed between evaluated systems.

---

## 5.9.1 Candidate Models

Based on the observations from Phases 1 and 2, two candidate deployment configurations were selected for large-scale validation.

### Configuration A

**Gemma-3-4B**

Reasoning Suppressed

This configuration demonstrated one of the strongest performances during the smaller benchmark and exhibited excellent formatting consistency after reasoning suppression.

### Configuration B

**Qwen2.5-3B-Instruct**

Unrestricted Reasoning

Although unrestricted reasoning produced slightly higher inference latency, Qwen consistently demonstrated stronger semantic quality and more stable behaviour across diverse legal queries.

Large-scale evaluation was therefore required to determine whether the trends observed during exploratory experiments generalized to a substantially larger benchmark.

---

# 5.9.2 Large-Scale Evaluation Results

> **INSERT TABLE 8 HERE**

**Large-scale LegalBench-RAG evaluation**

*(Directly use the complete Phase 3 table from the Master Evaluation Report.)*

The results reveal a striking contrast between the two evaluated deployment configurations.

While Gemma demonstrated highly competitive performance during the earlier exploratory benchmark, its behaviour deteriorated substantially during large-scale evaluation. The model experienced severe formatting instability, resulting in an error rate approaching complete evaluation failure.

In contrast, Qwen maintained consistent response generation across the entire benchmark while preserving high retrieval effectiveness and semantic similarity.

These observations demonstrate the importance of validating candidate deployment configurations using sufficiently large evaluation sets rather than relying exclusively on small exploratory benchmarks.

---

## 5.9.3 Analysis of Gemma Failure

One of the most important findings of this study concerns the behaviour of Gemma during large-scale evaluation.

Although Gemma initially appeared to be one of the strongest candidate models following reasoning suppression, its performance deteriorated dramatically when evaluated across the complete benchmark.

The primary cause of failure was not retrieval quality.

Retrieval performance remained essentially unchanged because both evaluated configurations employed the same embedding model and retrieval pipeline.

Instead, the degradation originated almost entirely from response generation.

Gemma produced a substantially larger number of malformed JSON outputs, resulting in a dramatic increase in formatting failures despite the presence of the JSON repair pipeline.

Consequently, only a very small proportion of benchmark queries produced valid responses suitable for downstream evaluation.

This observation highlights an important practical consideration.

Small benchmark evaluations may not adequately reveal deployment reliability.

A language model that performs well on tens of benchmark questions may nevertheless become unsuitable for production deployment if formatting stability deteriorates during sustained inference.

---

## 5.9.4 Selection of the Final Deployment Model

Unlike Gemma, Qwen demonstrated remarkably stable behaviour throughout the complete benchmark.

The final deployment configuration achieved

- Recall@5 = **100.0%**
- MRR = **1.000**
- BERTScore F1 = **0.8460**
- Corpus BLEU = **14.02**
- Corpus chrF = **31.54**
- Token F1 = **29.0%**
- Error Rate = **1.0%**

These results indicate that retrieval remained consistently successful while semantic similarity between generated explanations and benchmark answers remained high.

Although lexical metrics such as BLEU and Token F1 appear comparatively modest, these values should be interpreted alongside BERTScore and faithfulness evaluation.

Unlike benchmark answers that frequently reproduce contractual language verbatim, CivicSense AI intentionally generates simplified natural-language explanations designed for non-expert users.

Consequently, lexical overlap metrics underestimate explanation quality, whereas semantic evaluation more accurately reflects practical usefulness.

Considering retrieval quality, semantic similarity, response stability, and deployment reliability together, **qwen2.5-3b-instruct** was selected as the final production model for CivicSense AI.

---

# 5.10 Security Evaluation

Robustness against prompt injection constitutes an important requirement for Retrieval-Augmented Generation systems because retrieved documents are incorporated directly into the language model context.

To evaluate the effectiveness of the proposed Instruction Integrity mechanism, CivicSense AI was tested using the TeleAI-Safety benchmark.

The benchmark consists of fifty adversarial prompts specifically designed to manipulate language model behaviour through malicious instructions embedded within retrieved content.

Unlike traditional jailbreak attacks targeting user prompts, these attacks evaluate whether malicious document content can override trusted system instructions during Retrieval-Augmented Generation.

---

## 5.10.1 Evaluation Results

> **INSERT TABLE 9 HERE**

**TeleAI-Safety Evaluation**

*(Master Evaluation Report.)*

The proposed framework achieved an

**Attack Success Rate (ASR) of 0.0%**

across all fifty benchmark attacks.

Every evaluated attack resulted in successful refusal or safe behaviour.

No benchmark prompt successfully manipulated the language model into violating system instructions.

These results demonstrate that the Instruction Integrity mechanism effectively separates trusted system instructions from untrusted retrieved evidence while preserving normal Retrieval-Augmented Generation behaviour.

---

# 5.11 Discussion

The experimental results presented throughout this study provide several important insights regarding the design of trustworthy local Retrieval-Augmented Generation systems.

First, retrieval quality depends primarily on the embedding model and chunking strategy rather than the language model itself. Once an appropriate retrieval pipeline has been established, differences between language models emerge primarily in semantic quality, formatting robustness, latency, and reasoning behaviour.

Second, reasoning suppression does not universally improve Retrieval-Augmented Generation performance.

Instead, its effectiveness depends strongly upon the underlying language model.

Llama benefited substantially from suppressing reasoning, Gemma initially demonstrated considerable improvement, whereas Qwen consistently performed better using unrestricted reasoning.

These observations suggest that prompting strategies should be optimized individually for each language model rather than applied uniformly across architectures.

Third, semantic evaluation proved substantially more informative than lexical evaluation for explainable legal question answering.

Although benchmark answers frequently consist of verbatim legal clauses, CivicSense AI intentionally generates simplified explanations designed for readability.

Consequently, metrics such as BERTScore and LLM-as-a-Judge faithfulness more accurately capture explanation quality than Exact Match or BLEU alone.

Finally, large-scale validation revealed that response stability represents an essential deployment criterion.

Gemma appeared highly competitive during exploratory evaluation but proved unsuitable for production deployment because formatting failures increased dramatically during large-scale benchmarking.

Conversely, Qwen maintained consistent performance across the entire benchmark, ultimately making it the preferred production model despite not achieving the highest score in every individual metric during earlier experiments.

These findings emphasize the importance of comprehensive engineering evaluation when selecting lightweight language models for Retrieval-Augmented Generation.

Rather than optimizing a single evaluation metric, practical deployment requires balancing retrieval effectiveness, semantic quality, formatting robustness, latency, explainability, and security simultaneously.

---

# 6 Conclusion

This paper presented **CivicSense AI**, a fully local Explainable Retrieval-Augmented Generation framework for understanding legal and civic documents through evidence-grounded natural language explanations.

Unlike conventional conversational AI systems that rely primarily on pre-trained knowledge, CivicSense AI combines semantic retrieval with lightweight locally deployed language models to generate responses explicitly grounded in retrieved document evidence. The proposed framework additionally incorporates evidence attribution, instruction-integrity safeguards, and structured output validation to improve transparency, robustness, and practical usability.

Rather than selecting system components empirically, this work adopted a systematic engineering methodology. Embedding models, chunk-size configurations, lightweight language models, prompting strategies, and security mechanisms were independently evaluated through controlled experiments before integration into the final deployment.

Experimental evaluation using the LegalBench-RAG benchmark demonstrated that the selected deployment configuration achieved perfect retrieval performance while maintaining strong semantic similarity, low formatting error rates, and robust resistance against prompt injection attacks. The reasoning-ablation experiments further revealed that reasoning suppression affects lightweight language models differently, highlighting the importance of model-specific prompt engineering rather than universal prompting strategies.

Beyond presenting a practical legal assistant, this work contributes an empirical study of engineering trade-offs involved in deploying trustworthy local Retrieval-Augmented Generation systems on commodity consumer hardware. The findings indicate that reliable, explainable, and privacy-preserving legal document understanding can be achieved without relying on proprietary cloud-based language models.

Future work will investigate hybrid retrieval strategies, reranking models, multilingual legal document understanding, multimodal document processing, and larger local language models while maintaining explainability and efficient local deployment. Additional research will also explore adaptive retrieval techniques, structured knowledge integration, and broader evaluation across diverse legal jurisdictions to further improve the applicability of Retrieval-Augmented Generation for real-world legal assistance.