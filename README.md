# DSCI_575_project_gaultian_chrchow

Authors: Christine Chow and Ian Gault

https://github.com/UBC-MDS/DSCI_575_project_gaultian_chrchow

### Project Title/Why/Goal



### Dataset Description



### Setup Instructions

1. After opening a terminal, clone the repository:
    ```bash
    git clone https://github.com/UBC-MDS/DSCI_575_project_gaultian_chrchow.git
    cd DSCI_575_project_gaultian_chrchow
    ```

2. Create the environment:
    ```bash
    conda env create -f environment.yml
    ```

3. Activate the environment:
    ```bash
    conda activate 575_proj
    ```

4. Prepare Data and Search Index (Optional):

   There is preprocessed data and index available in the repository for immediate use. To regenerate them:
   * run all cells in `notebooks/milestone1_exploration.ipynb` to generate `merged.parquet`
   * run indexing script to generate `m25_index.pkl`
   ```bash
   python src/build_index.py
   ``` 

5. Setup environment variables:

    Create an `.env` file in root directory and add the following lines as needed:
    ```bash
    HF_TOKEN=<huggingface_token>
    ANTHROPIC_API_KEY=<your_key_here>  # Required for future milestones only
    ```

7. Run Streamlit app:
    ```bash
    streamlit run app/app.py
    ```
