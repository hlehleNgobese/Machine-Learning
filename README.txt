================================================================================
 INTEGRATED MACHINE LEARNING SYSTEMS PROJECT — README
================================================================================

PROJECT OVERVIEW
----------------
This project implements two intelligent systems:
  Component A: Hybrid RL + Ensemble Learning for SA Traffic Accident Prediction
  Component B: LLM + RAG System for SA Parliamentary Hansard Transcript Analysis


HOW TO RUN
----------
1. Open a terminal in the project root directory.
2. Install dependencies:
       pip install -r requirements.txt
3. Open the Jupyter notebook:
       jupyter notebook main_notebook.ipynb
4. Run all cells from top to bottom (Cell > Run All).

   NOTE: If you have the actual datasets, place them in the data/ folder and
   update the file paths in the notebook where indicated. Otherwise, the code
   generates synthetic data for demonstration.


PYTHON VERSION
--------------
Python 3.9 or higher (tested on Python 3.10)


DEPENDENCIES
------------
Core:
  pandas >= 1.5.0
  numpy >= 1.23.0
  matplotlib >= 3.6.0
  seaborn >= 0.12.0
  scikit-learn >= 1.2.0
  xgboost >= 1.7.0

NLP / Transformers (Component B):
  transformers >= 4.30.0
  sentence-transformers >= 2.2.0
  torch >= 2.0.0
  faiss-cpu >= 1.7.0

Install all at once:
  pip install pandas numpy matplotlib seaborn scikit-learn xgboost transformers sentence-transformers torch faiss-cpu


EXPECTED RUNTIME
----------------
Component A (Ensemble + RL):       ~1-2 minutes
Component B (NLP + RAG):           ~3-5 minutes (with transformer models)
Component B (fallback, no GPU):    ~1-2 minutes (uses TF-IDF + sklearn)
Total:                             ~5-10 minutes


HARDWARE REQUIREMENTS
---------------------
Minimum:
  - CPU: Any modern processor (Intel i5 / AMD Ryzen 5 or equivalent)
  - RAM: 8 GB
  - Disk: 2 GB free space (for model downloads)

Recommended (for BERT fine-tuning):
  - GPU: NVIDIA GPU with CUDA support (e.g., GTX 1060 or better)
  - RAM: 16 GB
  - The code includes CPU fallbacks if GPU is unavailable.


PROJECT STRUCTURE
-----------------
Machine-Learning/
  main_notebook.ipynb      <- Main Jupyter notebook (run this)
  component_a.py           <- Component A: Ensemble + RL code
  component_b.py           <- Component B: LLM + RAG code
  report.md                <- Professional report (export to PDF)
  requirements.txt         <- Python dependencies
  README.txt               <- This file
  data/
    preprocessed_accidents.csv   <- Generated after running Component A
    preprocessed_hansard.csv     <- Generated after running Component B
    confusion_matrices.png       <- Saved plots
    feature_importance.png
    rl_results.png
    sentiment_confusion_matrix.png


DATASETS
--------
Component A: South African Traffic Accident Dataset
  Source: https://www.kaggle.com/datasets/velile/car-accidents
  Place the CSV file in data/ folder and update the filepath in the notebook.

Component B: South African Parliamentary Hansard Transcripts
  Source: https://www.parliament.gov.za/hansard-papers?sorts[date]=-1
  Download transcripts, save as CSV, and update the filepath in the notebook.

If datasets are not available, the code generates representative synthetic data.


NOTES
-----
- The notebook is designed to run end-to-end without errors.
- If transformer libraries are not installed, the code falls back to
  scikit-learn equivalents automatically.
- All plots are saved to the data/ folder and displayed inline.
- Export report.md to PDF using any markdown editor or:
      pip install grip
      grip report.md --export report.pdf

================================================================================
