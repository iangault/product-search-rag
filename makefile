# Adapted to this project from a previous project done in 532 using codex

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
CYAN := \033[0;36m
RESET := \033[0m

.PHONY: help create prune process build-ir build-rag build-eval-set eval-retrieval upload-data run all clean

# Help command to list available commands
help:
	@echo -e "$(CYAN)Available commands:$(RESET)"
	@echo -e "  $(YELLOW)make create$(RESET)       - Create conda environment from environment.yml"
	@echo -e "  $(YELLOW)make prune$(RESET)        - Update conda environment from environment.yml with pruning"
	@echo -e "  $(YELLOW)make process$(RESET)      - Import raw data, process it, and save processed parquet files"
	@echo -e "  $(YELLOW)make build-ir$(RESET)     - Build and save BM25 and semantic retrieval indexes"
	@echo -e "  $(YELLOW)make build-rag$(RESET)    - Build and save the RAG chunk index"
	@echo -e "  $(YELLOW)make build-eval-set$(RESET) - Build pooled retrieval candidates for manual labeling"
	@echo -e "  $(YELLOW)make eval-retrieval$(RESET) - Run retrieval evaluation with precision@k, recall@k, and MRR"
	@echo -e "  $(YELLOW)make upload-data$(RESET)   - Upload processed data and indexes to Hugging Face (needs write token)"
	@echo -e "  $(YELLOW)make run$(RESET)          - Run the Streamlit app locally"
	@echo -e "  $(YELLOW)make all$(RESET)          - Clean artifacts, process data, build IR and RAG indexes, then run the app"
	@echo -e "  $(YELLOW)make clean$(RESET)        - Remove processed data and saved retrieval artifacts"

# Create conda environment
create:
	@echo -e "$(GREEN)Creating conda environment from environment.yml...$(RESET)"
	@conda env create -f environment.yml
	@echo -e "$(GREEN)Environment creation complete.$(RESET)"

# Update/prune conda environment
prune:
	@echo -e "$(CYAN)Updating conda environment from environment.yml with pruning...$(RESET)"
	@conda env update --file environment.yml --prune
	@echo -e "$(GREEN)Environment update complete.$(RESET)"

process:
	@echo -e "$(CYAN)Importing and processing raw data...$(RESET)"
	@python src/import_process.py
	@echo -e "$(GREEN)Processed parquet build complete.$(RESET)"

# Build and save BM25 and semantic retrieval indexes
build-ir:
	@echo -e "$(CYAN)Building BM25 and semantic indexes...$(RESET)"
	@python src/build_index.py
	@echo -e "$(GREEN)IR index build complete.$(RESET)"

build-rag:
	@echo -e "$(CYAN)Building RAG index...$(RESET)"
	@python src/build_rag.py
	@echo -e "$(GREEN)RAG index build complete.$(RESET)"

build-eval-set:
	@echo -e "$(CYAN)Building pooled retrieval label set...$(RESET)"
	@python src/build_retrieval_labels.py
	@echo -e "$(GREEN)Retrieval label set build complete if no labels file was already present.$(RESET)"

eval-retrieval:
	@echo -e "$(CYAN)Running retrieval evaluation...$(RESET)"
	@python src/evaluate_retrieval.py
	@echo -e "$(GREEN)Retrieval evaluation complete.$(RESET)"

upload-data:
	@echo -e "$(CYAN)Uploading processed data and indexes to Hugging Face...$(RESET)"
	@python src/hf_data.py
	@echo -e "$(GREEN)Upload complete.$(RESET)"

# Run the Streamlit app
run:
	@echo -e "$(GREEN)Running Streamlit app locally...$(RESET)"
	@streamlit run app/app.py

# Full workflow
all: clean process build-ir build-rag run

# Clean processed data and saved retrieval artifacts
clean:
	@echo -e "$(YELLOW)Cleaning processed data and retrieval artifacts...$(RESET)"
	@rm -f data/processed/merged.parquet
	@rm -f data/processed/processed.parquet
	@rm -f data/processed/bm25_index.pkl
	@rm -f data/processed/semantic.ids.npy
	@rm -f data/processed/semantic.index
	@rm -rf data/processed/rag_faiss
	@echo -e "$(GREEN)Clean complete.$(RESET)"
