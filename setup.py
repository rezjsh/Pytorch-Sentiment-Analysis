from setuptools import setup, find_packages

try:
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "A Pytorch-based Sentiment Analysis project using different NLP models."

__version__ = "0.0.1" 

REPO_NAME = "Pytorch-Sentiment-Analysis"
AUTHOR_USER_NAME = "rezjsh"
SRC_REPO = "sentiment_analysis" 
AUTHOR_EMAIL = "your.email@example.com" 

INSTALL_REQUIRES = [
    # Core DL/ML/Data
    "torch>=2.0.0",
    "torchmetrics>=1.0.0",
    "scikit-learn>=1.3.0",
    "joblib>=1.3.0",
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    
    # NLP & Transformers
    "transformers>=4.30.0",
    "datasets>=2.14.0",
    "sentence-transformers>=2.2.0",
    
    # Utilities & Visualization
    "tqdm>=4.66.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "PyYAML>=6.0", # Used for config files
]

# --- 3. Setup Call ---
setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A modular PyTorch Sentiment Analysis project featuring ML, DL, and Transformer models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    
    # --- Package Discovery Improvement ---
    package_dir={"": "src"}, # Tells setuptools that packages are found inside the 'src' directory
    packages=find_packages(where="src"), # Looks for packages inside the 'src' directory (e.g., src/models, src/data)
    
    # --- Dependency Injection ---
    install_requires=INSTALL_REQUIRES,
    
    # --- General Metadata ---
    python_requires=">=3.8", # Updating to 3.8+ is common practice now
)