# CellposeWrapper.jl

A robust Julia wrapper for [Cellpose v4](https://www.cellpose.org/), designed for high-performance biological image segmentation.

This wrapper provides a seamless interface between Julia and Python, with specific optimizations for **Apple Silicon** using MPS acceleration and **NVIDIA GPUs** using CUDA.

## 🚀 Features

- **Universal Hardware Support:** Automatically detects and uses `MPS` (Mac) or `CUDA` (Linux/Windows).
- **Cellpose v4 Compatibility:** Handles the new API structure automatically.
- **Advanced Parameter Control:** Full access to `flow_threshold`, `cellprob_threshold`, `augment`, and more directly from Julia.

---

## 🛠 Installation

Follow these steps to set up the environment from scratch.

### 1. Clone the Repository

Open your terminal and clone the folder:

```bash
git clone [https://github.com/diegocividini/CellposeWrapper.jl.git](https://github.com/diegocividini/CellposeWrapper.jl.git)
cd CellposeWrapper.jl
```
Ecco il contenuto completo per il file README.md.

Ho strutturato il documento in modo professionale, includendo tutte le istruzioni per l'installazione dell'ambiente Python, il collegamento con Julia, il download manuale dei modelli (necessario per il fix che abbiamo fatto) e gli esempi di utilizzo ottimizzati.

Copia tutto il blocco sottostante e incollalo nel file README.md nella root del tuo progetto.

Markdown

# CellposeWrapper.jl

A high-performance Julia wrapper for [Cellpose v4](https://www.cellpose.org/), designed for biological image segmentation.

This package provides a seamless interface between Julia and Python, with specific optimizations for **Apple Silicon (M1/M2/M3)** using MPS acceleration and **NVIDIA GPUs** using CUDA.

## 🚀 Key Features
- **Universal Hardware Support:** Automatically detects and uses `MPS` (Mac), `CUDA` (Linux/Windows), or CPU.
- **Cellpose v4 Compatibility:** Handles the new Python API structure automatically.
- **Tissuenet Legacy Support:** Includes logic to use the highly accurate legacy Tissuenet (TN2) weights instead of the newer (but sometimes less precise) `cpsam` models.
- **Advanced Parameter Control:** Full access to `flow_threshold`, `cellprob_threshold`, `augment`, `min_size` and `invert` directly from Julia.

---

## 🛠 Installation Guide

Follow these steps to set up the environment from scratch.

### 1. Clone the Repository
```bash
git clone [https://github.com/IL_TUO_USERNAME/CellposeWrapper.jl.git](https://github.com/IL_TUO_USERNAME/CellposeWrapper.jl.git)
cd CellposeWrapper.jl
```
### 2. Python Environment Setup
You need to create a local Python environment to handle Cellpose dependencies without conflicting with your system.

Recommended: `Python 3.10`.

#### Using `venv` (built-in Python tool)

```bash
# 1. Navigate to the python dependencies folder
cd deps/python

# 2. Create a virtual environment named .venv
# (If using pyenv, ensure local python is set to 3.10 first)
python3 -m venv .venv

# 3. Activate the environment
# Mac/Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

# 4. Install required packages
pip install -r requirements.txt

# 5. Return to the project root
cd ../..
```

### 3. Julia Environment Setup

Now you need to tell Julia to use the specific Python environment you just created.

Launch Julia in the project root (`julia`) and run:

```julia
import Pkg
Pkg.activate(".")

# 1. Install Julia dependencies defined in Project.toml
Pkg.instantiate()

# 2. Link Julia to your local Python .venv (Crucial Step!)
# This ensures Julia uses the correct libraries (Torch, Cellpose, etc.)
ENV["PYTHON"] = joinpath(pwd(), "deps", "python", ".venv", "bin", "python")

# 3. Build PyCall to lock this configuration
Pkg.build("PyCall")

# 4. Exit to apply changes
exit()
```

---

# ⚡️ Usage Examples

## 1. Basic Segmentation (Sparse Cells)
Use the default cyto3 model for standard cell cultures or sparse nuclei.

```julia

# Load the module (for local development)
include("src/CellposeWrapper.jl")
using .CellposeWrapper

# Run segmentation
# diameter=nothing lets Cellpose estimate cell size automatically
masks = CellposeWrapper.segment_image("assets/cells.jpg", diameter=nothing)

# Visualize results
CellposeWrapper.show_results(masks, "assets/cells.jpg")
```

## 2. Dense Tissue Segmentation (Advanced)
For histology images where cells touch each other (no gaps), use the tissuenet model with specific tuning.

```julia

include("src/CellposeWrapper.jl")
using .CellposeWrapper

masks = CellposeWrapper.segment_image(
    "assets/tissue_sample.jpg",
    diameter=25,       
    # Filter out noise/artifacts smaller than 100px
    min_size=100,           
    # High Precision Mode (4x slower, runs 4 rotations/flips)
    augment=true,
    # Increase sensitivity (lower threshold = more cells detected)
    cellprob_threshold=-0.5 
)

CellposeWrapper.show_masks(masks, "assets/tissue_sample.jpg")
```

---

A big thanks goes to [Josselin Morvan](https://github.com/sardinecan) who created the [version of this wrapper using SegmentAnything.jl](https://github.com/sardinecan/SegmentAnything.jl?tab=readme-ov-file). I used it as a reference to build this more specialized Cellpose wrapper.
