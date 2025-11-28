# README — HACK_aurigin.ai

# Aurigin.ai Hackathon – Audio Classification Baseline (2025)

This repository contains a clean and reconstructed baseline pipeline used for the  
Aurigin.ai Hackathon 2025.  
The project focuses on classifying short audio samples using MFCC feature extraction  
and a simple RandomForest baseline model.

---

## 📂 Project Structure

HACK_aurigin.ai/
├── src/
│ ├── main.py # Pipeline entry point
│ ├── hf_utils.py # Dataset loader + MFCC feature extraction
│ ├── hf_trainer.py # Training + submission utilities
│ ├── model.py # Model definitions
│ ├── trainer.py # Training helpers
│ └── utils.py # General utilities
├── requirements.txt
├── hackathon_plan.txt
└── .gitignore




Note:  
Local virtual environments, caches, large binary files, and audio data are intentionally  
ignored via `.gitignore`.

---

## 🚀 Setup & Installation

```bash
git clone https://github.com/aleksojdanic/HACK_aurigin.ai.git
cd HACK_aurigin.ai
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

```


##📌 Notes

 - The original dataset was private and is not included in this repository.
 - No virtual environments, caches, or large files are tracked.
 - This repository is meant as a clean baseline for further development.

##📝 License

Created for the Aurigin.ai Hackathon 2025.
Free to use and adapt.

