# CellposeWrapper.jl

[![CI](https://github.com/diegocividini/CellposeWrapper.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/diegocividini/CellposeWrapper.jl/actions)
![Julia](https://img.shields.io/badge/julia-1.10%20%7C%201.11-blueviolet)
![OS](https://img.shields.io/badge/os-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A lightweight and robust Julia wrapper for **Cellpose v4**, designed for biological image segmentation with optional visualization utilities.

This package provides a clean Julia API on top of the Python Cellpose backend, with **lazy initialization**, **automatic GPU detection**, and a **modular design** suitable for CI and production workflows.

---

## 🚀 Features

- **Cellpose v4 compatible**
  - Uses the new `CellposeModel` API
  - Supports **`cpsam`** (the only model type supported in v4)
- **Automatic hardware detection**
  - Apple Silicon (MPS)
  - NVIDIA GPUs (CUDA)
  - CPU fallback
- **Lazy Python initialization**
  - Python, Torch and Cellpose are loaded only when needed
- **Pure segmentation core**
  - No plotting or image dependencies required
- **Optional visualization via Julia extensions**
  - Visualization utilities are loaded only if plotting packages are installed
- **CI-safe**
  - Tests pass even when Python / Cellpose are not installed

---

## 🛠 Installation

### Julia package

```julia
using Pkg
Pkg.add("CellposeWrapper")
```

For local development:

```julia
Pkg.develop(path="path/to/CellposeWrapper.jl")
```

### 🐍 Python environment (required for segmentation)

Cellpose runs in Python.
You must provide a Python environment with:

`cellpose >= 4`

`torch`

`opencv-python`

Recommended: **Python 3.10**

Example using `venv`

```bash
cd deps/python
python3 -m venv .venv
source .venv/bin/activate

pip install cellpose torch opencv-python
```

Then tell Julia to use this Python:

```julia
using Pkg

# If you are using UNIX-like OS, the path is:
ENV["PYTHON"] = joinpath(pwd(), "deps", "python", ".venv", "bin", "python")
# If you are on Windows, the path is:
ENV["PYTHON"] = joinpath(pwd(), "deps", "python", ".venv", "Scripts", "python.exe")

Pkg.build("PyCall")
```

Restart Julia after building PyCall.

---

## ⚡️ Usage

### Basic segmentation

```julia
using Pkg 
Pkg.activate(".") # if you are not already in the package environment Pkg.activate("Path/To/CellposeWrapper.jl")
Pkg.instantiate()

using CellposeWrapper

res = segment_image("cells.png"; diameter=nothing)

masks = res.masks
```

Cellpose automatically:

- detects available hardware (CUDA / MPS / CPU)
- estimates the cell diameter if diameter = nothing
- uses the cpsam model

> ℹ️ If `model_type` is set to anything other than "cpsam", a warning is logged because Cellpose v4 currently supports only `cpsam`.

> 💡 `init!()` can be called explicitly to warm up Python and load models ahead of time, but is not required.

### Advanced parameters

```julia
res = segment_image(
    "tissue.png";
    diameter=25,
    min_size=100,
    augment=true,
    cellprob_threshold=-0.5
)
```

---

## 🎨 Visualization (optional)

Visualization utilities are **not loaded by default**.

> In order to visualize the results, you have to activate the environment and produce the segmentation results first, as shown in the previous section.

To enable them, load the required packages:

```julia
using Plots, Colors, FileIO, Images
```

This automatically activates the extension `CellposeWrapperVizExt`.

### Show results

```julia
CellposeWrapper.show_results(res, "cells.png"; view="masks")
```

Available views:

- `"masks"` – colored instance segmentation overlay
- `"flows"` – Cellpose flow visualization (requires `return_flows=true`)
- `"prob"` – cell probability map (requires `return_flows=true`)
- `"image"` – original image only

`show_results` displays plots and returns `nothing`.

#### Flows / Probability maps

```julia
res = segment_image("cells.png"; return_flows=true)

CellposeWrapper.show_results(res, "cells.png"; view="flows")
CellposeWrapper.show_results(res, "cells.png"; view="prob")
```

---

## 🧪 Testing & CI

The test suite is designed to be **CI-safe**:

Core tests do **not** require Python

Visualization tests run only if plotting dependencies are installed

Runtime Cellpose tests run **only if Python + Cellpose are available**

Run tests locally:

```julia
Pkg.activate(".")
Pkg.test()
```

---

## 📎 Design philosophy

- Segmentation-first API
- Visualization is optional and modular
- No hard dependency on plotting or image IO
- Safe for Julia General Registry and automated CI

> CellposeWrapper automatically uses CUDA when available via PyTorch.
> CUDA support depends on the presence of NVIDIA proprietary drivers and a CUDA-enabled PyTorch installation.
> If CUDA is not available, the wrapper falls back to CPU or Apple MPS automatically.

## 🙏 Acknowledgements

Inspired by Julia wrappers such as [SegmentAnything.jl](https://github.com/sardinecan/SegmentAnything.jl) and by the original [Cellpose](https://github.com/MouseLand/cellpose) project.
