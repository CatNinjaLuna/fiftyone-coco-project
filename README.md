# FiftyOne COCO Project

This project uses **FiftyOne** with a **micromamba environment** for COCO dataset loading and visualization.

## Prerequisites

The following setup has been completed:

- ✅ Micromamba is installed
- ✅ Shell is initialized for zsh
- ✅ Environment `fiftyone` exists with FiftyOne installed

## Getting Started

### Activating the Environment

**Option 1: Terminal activation (recommended)**

```bash
micromamba activate fiftyone
```

Verify the installation:

```bash
python -c "import fiftyone as fo; print(fo.__version__)"
```

Run the COCO loader:

```bash
python load_coco_fiftyone.py
```

**Option 2: One-liner (no activation required)**

```bash
micromamba run -n fiftyone python load_coco_fiftyone.py
```

### VS Code Python Interpreter

To avoid `ModuleNotFoundError`, configure VS Code to use the correct Python interpreter:

1. Press `Cmd + Shift + P` (macOS) or `Ctrl + Shift + P` (Windows/Linux)
2. Select **Python: Select Interpreter**
3. Choose: `~/.local/share/mamba/envs/fiftyone/bin/python`

## Troubleshooting

| Issue                                             | Solution                                                                                                                                                                                                         |
| ------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Terminal prompt shows `(base)`                    | Wrong environment activated. Run `micromamba activate fiftyone`                                                                                                                                                  |
| `ModuleNotFoundError: No module named 'fiftyone'` | **Always activate the fiftyone environment before running the script!** Use `micromamba activate fiftyone` then `python load_coco_fiftyone.py`, OR use `micromamba run -n fiftyone python load_coco_fiftyone.py` |
| `import fiftyone` fails                           | VS Code interpreter not set to `fiftyone` environment                                                                                                                                                            |
| Changes don't take effect                         | Restart VS Code after environment or shell changes                                                                                                                                                               |

### Important: Running the Script

To avoid module import errors, **always run the script within the fiftyone environment**:

**Method 1: Activate first (recommended)**

```bash
micromamba activate fiftyone
python load_coco_fiftyone.py
```

**Method 2: Use micromamba run**

```bash
micromamba run -n fiftyone python load_coco_fiftyone.py
```

## Project Structure

```
.
├── load_coco_fiftyone.py    # Main script to load COCO dataset
├── README.md                 # This file
└── quickstart/               # Dataset directory
    ├── info.json
    ├── metadata.json
    ├── samples.json
    └── data/
```
