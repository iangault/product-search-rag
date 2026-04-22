# DSCI 575 Project: Amazon Product Search Assistant

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](environment.yml)
[![Built with Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-ff4b4b.svg)](https://streamlit.io/)

Authors: Christine Chow and Ian Gault

GitHub Repository: [UBC-MDS/DSCI_575_project_gaultian_chrchow](https://github.com/UBC-MDS/DSCI_575_project_gaultian_chrchow)

### Project Goal

This project builds a context-aware product search assistant using the Amazon Reviews 2023 dataset. The app supports BM25 keyword search, semantic vector search, hybrid search, and retrieval-augmented generation (RAG) workflows including a hybrid RAG mode for grounded question answering over product metadata and review text.

### Dataset Description

For this project, we are using the Appliances category from the [Amazon Reviews 2023 dataset](https://amazon-reviews-2023.github.io) collected in 2023 by [McAuley Lab](https://cseweb.ucsd.edu/~jmcauley/). There are two different files available in this category: *User Reviews* containing reviews, star ratings, etc, and *Item Metadata* containing descriptions, price, and other features. The raw `.jsonl.gz` files are loaded, written to parquet, and merged together on `parent_asin` to create a searchable product-level corpus. BM25 query and document text are then tokenized with lowercasing, punctuation stripping, stopword removal, and stemming at retrieval time. Merging the files together allows the retrieval system to match queries from both the product specifications and user reviews.

### Retrieval Workflows

BM25 indexes `product_title`, `features`, `description`, `categories`, `details`, and `review_text`. Semantic search indexes `product_title`, `features`, `description`, `categories`, and `details`, excluding review text so the vector index stays focused on product metadata.

BM25 search was performed using the [`rank_bm25`](https://pypi.org/project/rank-bm25/) package. Both the corpus and user queries are processed and tokenized before results are ranked based on a derivation of the TF-IDF (term frequency - inverse document frequency).

Semantic search was executed using the `all-MiniLM-L6-v2` transformer model from the [`sentence-transformers`](https://huggingface.co/sentence-transformers) to generate document embeddings. Embedding generation automatically detects whether CUDA is available and otherwise falls back to CPU. These embeddings were then indexed using the CPU build of [FAISS](https://faiss.ai/index.html) (Facebook AI Similarity Search) to keep indexing and search portable across both CUDA-enabled machines and standard laptops. Cosine similarity was used to calculate vector similarity and results are retrieved using k-nearest neighbours.

### LLM Model

We use `Qwen3-32B` via the [Groq API](https://console.groq.com/) as the LLM for both RAG pipelines. Qwen3 is an open-source large language model from Alibaba Cloud that generates clear, grounded answers when given retrieved product context. We picked the 32B size over smaller variants for better answer quality, and Groq's free API is fast enough for interactive use without requiring local GPU.

### RAG Workflows

#### Semantic RAG

Semantic RAG uses a FAISS vector store built from chunked product documents to find the most relevant results for a query. The retrieved chunks are built into a context block and passed to the LLM to generate an answer.

```mermaid
flowchart TD
    P[processed.parquet] --> Q[build_documents<br/>src/utils.py]
    Q --> R[normalize + clean_html<br/>flatten text and strip HTML]
    R --> B[Saved FAISS retriever<br/>data/processed/rag_faiss]
    A[User Query] --> B
    B --> C[Top-5 retrieved chunks]
    C --> D[relevant_text:<br/>format chunk metadata + text]
    D --> E[build_prompt:<br/>query + grounded context]
    E --> F[Qwen3-32B via Groq API]
    F --> G[clean_response:<br/>remove think block]
    G --> H[Display answer +<br/>supporting source cards]
```

#### Hybrid RAG

Hybrid RAG combines BM25 and semantic search results using Reciprocal Rank Fusion (RRF) into a single ranked list of products. The top ranked products are built into a context block and passed to the LLM to generate an answer.

```mermaid
flowchart TD
    P[processed.parquet] --> Q[build_documents<br/>src/utils.py]
    Q --> R[normalize + clean_html<br/>flatten text and strip HTML]
    R --> B[BM25 index]
    R --> C[Semantic index]
    A[User Query] --> B
    A --> C
    B --> D[BM25Search.retrieve]
    C --> E[SemanticSearch.retrieve]
    D --> F[Top results by parent_asin]
    E --> G[Top results by parent_asin]
    F --> H[HybridRetriever:<br/>Reciprocal Rank Fusion]
    G --> H
    H --> I[Top-5 fused products]
    I --> J[relevant_text_hybrid:<br/>format product context]
    J --> K[build_prompt:<br/>query + grounded context]
    K --> L[Qwen3-32B via Groq API]
    L --> M[clean_response:<br/>remove think block]
    M --> N[Display answer +<br/>supporting product cards]
```

### Web App Features

To interact with the information retrieval systems, we developed a simple web app using Streamlit. Features include:

* Search Mode Selection: Users can toggle between BM25, Semantic, Hybrid, RAG and Hybrid RAG search modes

* User Query Input: A natural language text box for querying

* Detailed Search Results: For each retrieved product, the app displays the product title, a truncated review, the star rating, and the retrieval score for BM25 or Semantic search

* RAG Answer Panel: In RAG mode and Hybrid RAG modes, the app generates a grounded answer and shows supporting product cards. RAG mode also shows the retrieved chunk text used as evidence, while Hybrid RAG shows product review context.

* Out-of-scope query guardrail in RAG modes using LLM-based query filtering to reject queries unrelated to Amazon appliances, kitchen products, or product shopping.

### Repository Structure

BM25, semantic retrieval, and RAG components are defined in `src/` and built through separate indexing scripts.

```text
DSCI_575_project_gaultian_chrchow/
├── app/
│   └── app.py
│      Streamlit application for BM25, semantic, and RAG-based product search.
│
├── data/
│   ├── raw/
│   │   Raw parquet inputs created from the Amazon metadata and review source files.
│   │   - `meta_raw.parquet`
│   │   - `reviews_raw.parquet`
│   ├── eval/
│   │   Evaluation queries and manual relevance labels for retrieval evaluation.
│   │   - `retrieval_queries.json`
│   │   - `retrieval_labels.csv`
│   └── processed/
│       Processed data and saved retrieval indexes used by the app.
│       - `processed.parquet`
│       - `bm25_index.pkl`
│       - `semantic.index`
│       - `semantic.ids.npy`
│
├── src/
│   ├── bm25.py
│   │   BM25 retrieval class, including index building, searching, saving, and loading.
│   ├── import_process.py
│   │   Imports raw source data, merges metadata and reviews, and writes processed parquet files.
│   ├── semantic.py
│   │   Semantic retrieval class using SentenceTransformers and FAISS.
│   ├── build_index.py
│   │   Script to build and save BM25 and semantic search indexes from the processed dataset.
│   ├── build_rag.py
│   │   Script to build and save the FAISS index used by the RAG pipeline.
│   ├── hybrid.py
│   │   Hybrid retriever class combining BM25 and semantic search using Reciprocal Rank Fusion (RRF).
│   ├── rag_pipeline.py
│   │   RAG retrieval, prompt-building, and answer-generation helpers.
│   ├── build_retrieval_labels.py
│   │   Pools BM25, semantic, and hybrid results into a CSV for human relevance labeling.
│   ├── evaluate_retrieval.py
│   │   Quantitative retrieval evaluation for BM25, semantic, and hybrid search using human-labeled precision@k, recall@k, and MRR.
│   ├── prompts.py
│   │   Prompt template helpers for the RAG answer generation step.
│   └── utils.py
│       Utility helpers used by retrieval:
│       - `text_preprocessor`: tokenization, stopword removal, and stemming for BM25
│       - `clean_html`: strips HTML tags and normalizes display text
│       - `normalize`: converts strings, lists, and dicts into plain text
│       - `build_documents`: combines selected columns into one searchable document per row
│
├── notebooks/
│   Project notebooks for EDA, preprocessing, and experimentation.
│   - `milestone1_exploration.ipynb`
│   - `milestone2_exploration.ipynb`
│   - `milestone2_rag.ipynb`
│
├── results/
│   Project discussion notes, workflow diagrams, and retrieval evaluation outputs.
│   - `final_discussion.md`
│   - `milestone1_discussion.md`
│   - `milestone2_discussion.md`
│   - `retrieval_eval_results.json`
│
├── requirements.txt
│   Python package requirements.
│
├── environment.yml
│   Conda environment specification for reproducing the project setup.
│
├── makefile
│   Convenience commands for environment setup, preprocessing, indexing, evaluation, app launch, and cleanup.
│
├── Milestone_submission.md
│   Submission-specific project summary and milestone notes.
│
├── README.md
│   Project overview, setup instructions, and usage documentation.
│
└── LICENSE
    Project license.
```

### Installation and Setup

 **Note**: For a summary of makefile actions:

```bash
make help
```

1.  **Clone the repository** After opening a terminal, clone the repository and navigate to the project directory:

    ```bash
    git clone https://github.com/UBC-MDS/DSCI_575_project_gaultian_chrchow.git
    cd DSCI_575_project_gaultian_chrchow
    ```

2.  **Environment Setup**

    i. Create or Prune the Conda environment:


        make create
        make prune


    ii. Activate the Conda environment:


        conda activate 575_proj


3.  **Environment Variables** Create an `.env` file in root directory. Note: Do not commit this file to Github!

    ```bash
    GROQ_API_KEY=<your_groq_api_key>
    ```

4. **Data Preparation and Indexing** To reproduce our results:

    1. **Import and Pre-Processing**

        a. Import the raw metadata and review data from <https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw>
        
        b. Merge datasets
        
        c. Pre-processing of text data
        
        d. Save as `processed.parquet`

        To balance processing time with meaningful information retrievals, we chose to load 20k metadata products and 200k reviews.

        ```bash
        make process
        ```

    2. **EDA** Optional step. Internal data exploration to inform analysis

        a. Run all cells in `notebooks/milestone1_exploration.ipynb` for the earlier Milestone 1 exploration workflow.

        b. Run all cells in `notebooks/milestone2_exploration.ipynb` to explore `data/processed/processed.parquet` and confirm pre-processing steps taken in `import_process.py`. (Improvements made in Milestone 2)

    3.  **Build IR indexes:** run the indexing script to build BM25 and Semantic search indices:

        Derived from `data/processed/processed.parquet`.

        Due to the large file sizes, processed indexing artifacts are ignored by git, so `make build-ir` is an essential step to run BM25 and Semantic search in the app.

        ```bash
        make build-ir
        ```

    4.  **Build RAG index:** run the RAG indexing script to build the persisted FAISS chunk index used by the RAG mode:

        Also derived from `data/processed/processed.parquet`.

        `make build-rag` is required before using RAG mode in the app.

        ```bash
        make build-rag
        ```

5.  **Running the Web App** Launch the Streamlit dashboard:

    ```bash
    make run
    ```

    Once finished using the app, close the window in the browser, and in terminal press `ctrl + c` to stop running the app.

6. **Remove processed data and index artifacts** in `data/processed/`:

    If you want to return the project to a clean generated-data state before closing up the project.

    ``` bash
    make clean
    ```

To run the essential steps of the full pipeline `make clean`, `make process`, `make build-ir`, `make build-rag`, and `make run`:

``` bash
make all
```

7. **RAG Retrieval Evaluation**

    We used a custom script-based approach rather than a RADAS workflow because the goal here was to evaluate the information retrieval component of the RAG system without introducing an LLM into the labelling pipeline itself.

    1. **Build a manual retrieval labeling set:** pool top candidates from BM25, semantic, and hybrid search into a CSV for human relevance judgments.

        ```bash
        make build-eval-set
        ```

    This writes `data/eval/retrieval_labels.csv`. Manually fill the `relevant` column with `yes` or `no` for every pooled candidate row.

    This step is intended to create the labeling file before human review. If `retrieval_labels.csv` already exists, the script will stop rather than overwrite it. To intentionally replace the file (warning: do not do), run `python src/build_retrieval_labels.py --force`.

    2.  **Run retrieval evaluation:** after labeling every pooled candidate, evaluate BM25, semantic, and hybrid retrieval quantitatively with `precision@k`, `recall@k`, and `MRR`.

        ```bash
        make eval-retrieval
        ```

    This is the command to use for presenting quantitative evaluation results after the human judgments are complete. It reads the judged `data/eval/retrieval_labels.csv` file and writes summary and per-query metrics to `results/retrieval_eval_results.json`.

    Each of the 10 tested queries can be biased towards BM25 or Semantic interpretation. Therefore, the terminal-based output after running this command shows the average scores across quieries. This gives a more fair evaluation to the retrieval method itself.

    The summary results are shared in `final_discussion.md`.
