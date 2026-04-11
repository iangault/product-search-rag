# DSCI 575 Project: Amazon Product Search Assistant

Authors: Christine Chow and Ian Gault

GitHub Repository: [UBC-MDS/DSCI_575_project_gaultian_chrchow](https://github.com/UBC-MDS/DSCI_575_project_gaultian_chrchow)

### Project Goal

This project aims to build a context-aware product search assistant using the Amazon Reviews 2023 dataset with BM25 keyword and semantic vector searches. Future milestones will incorporate the use of LLMs in the search assistant.

### Dataset Description

For this project, we are using the Appliances category from the [Amazon Reviews 2023 dataset](https://amazon-reviews-2023.github.io) collected in 2023 by [McAuley Lab](https://cseweb.ucsd.edu/~jmcauley/). There are two different files available in this category: *User Reviews* containing reviews, star ratings, etc, and *Item Metadata* containing descriptions, price, and other features. The raw .json.gz files were loaded and merged together on `parent_asin` before undergoing text preprocessing (lowercasing and punctuation removal) to create a searchable corpus. Merging the files together would allow the retrieval system to match queries from both the product specifications and user reviews.

### Retrieval Workflows

BM25 search was performed using the [`rank_bm25`](https://pypi.org/project/rank-bm25/) package. Both the corpus and user queries are processed and tokenized before results are ranked based on a derivation of the TF-IDF (term frequency - inverse document frequency).

Semantic search was executed using the `all-MiniLM-L6-v2` transformer model from the [`sentence-transformers`](https://huggingface.co/sentence-transformers) to generate document embeddings. These embeddings were then indexed using [FAISS](https://faiss.ai/index.html) (Facebook AI Similarity Search) to enable similarity searches. Cosine similarity was used to calculate vector similarity and results are retrieved using k-nearest neighbours.

### Web App Features

To interact with the information retrieval systems, we developed a simple web app using Streamlit. Features include:

* Search Mode Selection: Users can toggle between BM25 and Semantic search methods to compare results

* User Query Input: A natural language text box for querying

* Detailed Search Results: For each retrieved product, the app will display the product title, a truncated review, the star rating and the retrieval score (BM25 or Semantic)

### Repository Installation

1.  **Installation** After opening a terminal, clone the repository and navigate to the project directory:
    ```bash
    git clone https://github.com/UBC-MDS/DSCI_575_project_gaultian_chrchow.git
    cd DSCI_575_project_gaultian_chrchow
    ```

### Setup Instructions

**Note**: For a summary of makefile actions:

```bash
make help
```

2.  **Environment Setup**

    1. Create or Prune the Conda environment:

    ``` bash
    make create
    make prune
    ```

    2. Activate the Conda environment:

    ```bash
    conda activate 575_proj
    ```

3.  **Environment Variables** Create an `.env` file in root directory. Note: Do not commit this file to Github!!!

    ``` bash
    HF_TOKEN=<your_huggingface_token>
    ANTHROPIC_API_KEY=<your_anthropic_api_key_here>
    ```

4. **Data Preparation and Indexing** To reproduce our results:

    1.  **EDA** run all cells in `notebooks/milestone1_exploration.ipynb` to process raw data into `data/processed/merged.parquet`.

    2.  **Indexing:** run the indexing script in to build BM25 and Semantic search indices:

        ``` bash
        make build
        ```

        Note: Preprocessed indices may already be available in `data/processed/` for immediate use.

        **Remove index data** in `data/processed/`:

        ``` bash
        make clean
        ```

5.  **Running the Web App** Launch the Streamlit dashboard:

    ```bash
    make run
    ```
