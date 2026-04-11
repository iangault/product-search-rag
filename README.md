# DSCI_575_project_gaultian_chrchow

Authors: Christine Chow and Ian Gault

https://github.com/UBC-MDS/DSCI_575_project_gaultian_chrchow

### Project Title/Why/Goal



### Dataset Description



### Setup Instructions

1. After opening a terminal, clone the repository:
```bash
git clone https://github.com/UBC-MDS/DSCI_575_project_gaultian_chrchow.git
```

2. Create the environment:
```bash
conda env create -f environment.yml
```

3. Activate the environment:
```bash
conda activate 575_proj
```

4. Setup environment variables:

    Create a `.env` file in root directory and add the following lines, as needed:
    ```bash
    HF_TOKEN=...
    ANTHROPIC_API_KEY=...  # Last milestone only
    ```

5. Run Streamlit app:
```bash
streamlit run app/app.py
```
