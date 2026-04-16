# Adapted to this project from a previous project done in 532 using codex

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
CYAN := \033[0;36m
RESET := \033[0m

.PHONY: help create prune process build run all clean

# Help command to list available commands
help:
	@echo -e "$(CYAN)Available commands:$(RESET)"
	@echo -e "  $(YELLOW)make create$(RESET)       - Create conda environment from environment.yml"
	@echo -e "  $(YELLOW)make prune$(RESET)        - Update conda environment from environment.yml with pruning"
	@echo -e "  $(YELLOW)make process$(RESET)      - Import raw data, process it, and save parquet files"
	@echo -e "  $(YELLOW)make build$(RESET)        - Build and save BM25 and semantic indexes"
	@echo -e "  $(YELLOW)make run$(RESET)          - Run the Streamlit app locally"
	@echo -e "  $(YELLOW)make all$(RESET)          - Clean artifacts, process data, build indexes, then run the app"
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

# Build and save retrieval indexes
build:
	@echo -e "$(CYAN)Building BM25 and semantic indexes...$(RESET)"
	@python src/build_index.py
	@echo -e "$(GREEN)Index build complete.$(RESET)"

# Run the Streamlit app
run:
	@echo -e "$(GREEN)Running Streamlit app locally...$(RESET)"
	@streamlit run app/app.py

# Full workflow
all: clean process build run

# Clean processed data and saved retrieval artifacts
clean:
	@echo -e "$(YELLOW)Cleaning processed data and retrieval artifacts...$(RESET)"
	@rm -f data/processed/processed.parquet
	@rm -f data/processed/bm25_index.pkl
	@rm -f data/processed/semantic.ids.npy
	@rm -f data/processed/semantic.index
	@echo -e "$(GREEN)Clean complete.$(RESET)"
