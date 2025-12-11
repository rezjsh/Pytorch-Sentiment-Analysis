import pathlib
import logging
import os # Added for path simplification

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- Define Project Name ---
# This name is used for the Python package (e.g., 'pip install sentiment-analysis')
PROJECT_SLUG = "sentiment_analysis" 

# --- Define Directories to Create ---
dirs_to_create = [
    f"configs",             # Centralized configuration
    f"data/raw",            # Raw data landing zone
    f"data/processed",      # Processed data output
    f"docs",
    f"logs",                # All execution logs
    f"notebooks",           # Jupyter notebooks
    f"reports/figures",     # EDA/Model plots
    f"checkpoints",         # Model checkpoints (.pt, .pkl)
    f"tests",               # Unit tests
    
    # --- Source Code Structure (PYTHON PACKAGE) ---
    f"src/{PROJECT_SLUG}", 
    f"src/{PROJECT_SLUG}/data",
    f"src/{PROJECT_SLUG}/features",
    f"src/{PROJECT_SLUG}/models",
    f"src/{PROJECT_SLUG}/utils",
    f"src/{PROJECT_SLUG}/evaluation",
    
    # MLOps Components (Standard structure)
    f"src/{PROJECT_SLUG}/components", 
    f"src/{PROJECT_SLUG}/pipeline",
    f"src/{PROJECT_SLUG}/config",    # Contains configuration reader logic (e.g., ConfigurationManager)
    f"src/{PROJECT_SLUG}/entity",    # Contains config_entity.py
    f"src/{PROJECT_SLUG}/constants", # Contains constants.py
]

# --- Define Files to Create ---
# NOTE: All files inside the src/{PROJECT_SLUG} directory must start with __init__.py
files_to_create = [
    # Config files (outside src)
    "configs/config.yaml",
    "configs/params.yaml",
    "configs/logging_config.yaml",
    
    # Root files
    "README.md",
    "setup.py",
    "requirements.txt",
    ".gitignore",
    "main.py",
    "Makefile",
    "Dockerfile",
    ".env.example",
    
    # Src Package Init Files
    f"src/{PROJECT_SLUG}/__init__.py",
    f"src/{PROJECT_SLUG}/data/__init__.py",
    f"src/{PROJECT_SLUG}/features/__init__.py",
    f"src/{PROJECT_SLUG}/models/__init__.py",
    f"src/{PROJECT_SLUG}/utils/__init__.py",
    f"src/{PROJECT_SLUG}/evaluation/__init__.py",
    f"src/{PROJECT_SLUG}/components/__init__.py",
    f"src/{PROJECT_SLUG}/pipeline/__init__.py",
    f"src/{PROJECT_SLUG}/config/__init__.py",
    f"src/{PROJECT_SLUG}/entity/__init__.py",
    f"src/{PROJECT_SLUG}/constants/__init__.py",
    
    # Component/Pipeline Files
    f"src/{PROJECT_SLUG}/components/data_ingestion.py",
    f"src/{PROJECT_SLUG}/components/data_validation.py",
    f"src/{PROJECT_SLUG}/components/data_transformation.py",
    f"src/{PROJECT_SLUG}/components/model_trainer.py",
    f"src/{PROJECT_SLUG}/components/model_evaluation.py",
    f"src/{PROJECT_SLUG}/pipeline/stage_01_data_ingestion.py",
    f"src/{PROJECT_SLUG}/pipeline/stage_02_data_validation.py",
    f"src/{PROJECT_SLUG}/pipeline/stage_03_data_transformation.py",
    f"src/{PROJECT_SLUG}/pipeline/stage_04_model_trainer.py",
    f"src/{PROJECT_SLUG}/pipeline/stage_05_model_evaluation.py",
    
    # Config/Entity Files
    f"src/{PROJECT_SLUG}/config/configuration.py",
    f"src/{PROJECT_SLUG}/entity/config_entity.py",
    f"src/{PROJECT_SLUG}/constants/constants.py",
    
    # Utility Files
    f"src/{PROJECT_SLUG}/utils/common.py", # Replaces logging_setup, helpers, file_io, exceptions
]

# --- Basic Gitignore Content ---
gitignore_content = """
# Standard Python ignores
__pycache__/
*.py[cod]
*.so

# Environments
.env
.venv
env/
venv/
environment.yml

# Data and Logs
data/
logs/
*.log

# Models/Checkpoints
checkpoints/
models/

# IDE files
.vscode/
.idea/

# Jupyter
.ipynb_checkpoints
"""

# --- Create Structure ---
logging.info(f"Starting project structure creation for package: {PROJECT_SLUG}")

# Create directories
for dir_path_str in dirs_to_create:
    path = pathlib.Path(dir_path_str)
    try:
        path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created directory (or verified exists): {path}")
    except OSError as e:
        logging.error(f"Failed to create directory {path}: {e}")

# Create files
for file_path_str in files_to_create:
    file_path = pathlib.Path(file_path_str)
    
    # Ensure parent directory exists before touching the file
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create file if it doesn't exist
    if not file_path.exists():
        try:
            file_path.touch()
            logging.info(f"Created file: {file_path}")
        except Exception as e:
            logging.error(f"Failed to create file {file_path}: {e}")
            continue

        # Add initial content to __init__.py files
        if file_path.name == "__init__.py":
            logging.info(f"Initialized package: {file_path}")
            
        # Add initial content to README.md
        if file_path.name == "README.md" and file_path.stat().st_size == 0:
            file_path.write_text(f"# {PROJECT_SLUG.replace('_', ' ').title()}\n", encoding='utf-8')
            logging.info(f"Added title to {file_path.name}")
            
# Create or update .gitignore in the root directory
gitignore_path = pathlib.Path(".gitignore")
try:
    existing_lines = set()
    if gitignore_path.exists():
        with open(gitignore_path, "r", encoding='utf-8') as f:
            existing_lines = set(line.strip() for line in f.read().splitlines() if line.strip())

    new_lines = [line.strip() for line in gitignore_content.strip().splitlines() if line.strip()]
    
    lines_to_add = [line for line in new_lines if line not in existing_lines]
    
    if lines_to_add:
        with open(gitignore_path, "a", encoding='utf-8') as f:
            f.write("\n") # Ensure separation from previous content
            f.write("\n".join(lines_to_add))
        logging.info(f"Updated .gitignore with {len(lines_to_add)} new lines.")
    else:
        logging.info(".gitignore file is up-to-date.")

except Exception as e:
    logging.error(f"Failed to handle .gitignore: {e}")

logging.info("Project structure creation process finished.")