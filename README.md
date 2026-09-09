# CORNEAL-TRUST

**Corneal Knowledge Graph Reasoning & Evidential Trust Index for Early Diabetic Peripheral Neuropathy Triage**

## Project Overview

CORNEAL-TRUST is a medical AI system for early detection of Diabetic Peripheral Neuropathy (DPN) using In-Vivo Corneal Confocal Microscopy (IVCCM) images. The system transitions from uncalibrated image classification to structured, knowledge-guided, uncertainty-aware clinical triage.

## Key Components

1. **Multi-Scale Feature Extraction** - Lightweight U-Net/CS-Net backbone for nerve skeletonization and topological feature extraction
2. **Corneal Knowledge Graph (CKG)** - Deterministic diagnostic reasoning and LLM guardrails
3. **Corneal Diagnostic Trust Index (CDTI)** - Unified reliability score with automated referral rejection
4. **Corneal Digital Twin** - Simulated edge environment with INT8 ONNX execution

## Quick Start

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
CORNEAL_TRUST/
├── src/                 # Source code
│   ├── data/           # Data loaders & preprocessing
│   ├── models/         # Neural network architectures
│   ├── knowledge_graph/ # Corneal Knowledge Graph
│   ├── digital_twin/   # Simulation environment
│   └── utils/          # Helper functions
├── configs/            # YAML configuration files
├── data/               # Dataset files
├── outputs/            # Models, logs, results
├── notebooks/          # Jupyter notebooks
├── scripts/            # Training/inference scripts
├── tests/              # Unit tests
└── docs/               # Documentation
```

## Author

**Gaurav Sahoo**
