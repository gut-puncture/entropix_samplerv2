# research-controller

AlphaZero-Style Macro-Action Controller for Text Generation.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd research-controller
    ```

2.  **Install Miniconda (if not already installed):**
    Follow the official installation guide: [https://docs.conda.io/projects/miniconda/en/latest/](https://docs.conda.io/projects/miniconda/en/latest/)

3.  **Create and activate the conda environment:**
    ```bash
    # Ensure you are in the research-controller directory
    conda create --name research-controller-env python=3.11 -y
    conda activate research-controller-env
    ```

4.  **Install dependencies:**
    ```bash
    # First install most packages via pip
    pip install -r requirements.txt
    
    # Then install sentencepiece via conda
    conda install sentencepiece=0.1.97 -c conda-forge -y
    ```
    *(Note: `kenlm` installation is currently deferred. It will be handled later.)*

5.  **Set OpenAI API Key (if needed for topic drift):**
    ```bash
    export OPENAI_API_KEY="sk-<YOUR-KEY>"
    ```

## Usage

*(To be added: instructions for training, evaluation, serving)*

## Development

*(To be added: instructions for running tests, etc.)* 