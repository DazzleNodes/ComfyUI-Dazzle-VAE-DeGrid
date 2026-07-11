# ComfyUI-Dazzle-VAE-DeGrid

A new project created from git-repokit-template

## Installation

```bash
pip install ComfyUI_Dazzle_VAE_DeGrid
```

### From Source

```bash
git clone https://github.com/DazzleNodes/ComfyUI-Dazzle-VAE-DeGrid.git
cd ComfyUI-Dazzle-VAE-DeGrid
pip install -e ".[dev]"
```

## Usage

```bash
ComfyUI-Dazzle-VAE-DeGrid --help
```

## Development

```bash
# Clone and install
git clone https://github.com/DazzleNodes/ComfyUI-Dazzle-VAE-DeGrid.git
cd ComfyUI-Dazzle-VAE-DeGrid
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -v

# Install git hooks (if using repokit-common submodule)
bash scripts/repokit-common/install-hooks.sh
```

## License

GPL-3.0-or-later. See [LICENSE](LICENSE) for details.

