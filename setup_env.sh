#!/bin/bash
set -e

# ==== cd-circuit-mechanism 环境搭建脚本 ====
CONDA_ENV_NAME="cd-circuit"
PYTHON_VERSION="3.10"
PIP_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
TORCH_INDEX="https://download.pytorch.org/whl/cu124"

# 1. 检测 conda
if [ -f "$HOME/miniconda3/bin/conda" ]; then
    export PATH="$HOME/miniconda3/bin:$PATH"
fi
if ! command -v conda &>/dev/null; then
    echo "ERROR: conda not found"; exit 1
fi
echo "[OK] conda: $(conda --version)"

# 2. 创建 conda 环境
if conda env list | grep -q "$CONDA_ENV_NAME"; then
    echo "[SKIP] env exists"
else
    conda create -n "$CONDA_ENV_NAME" python="$PYTHON_VERSION" -y
fi

# 3. 激活环境
export PATH="$HOME/miniconda3/envs/$CONDA_ENV_NAME/bin:$PATH"
echo "[OK] $(python --version) at $(which python)"

# 4. 安装 PyTorch (CUDA 12.4)
pip install "torch>=2.4,<2.6" "torchvision>=0.19,<0.21" "torchaudio>=2.4,<2.6" --index-url "$TORCH_INDEX" -q

# 5. 安装 ML 依赖
pip install transformer_lens sae_lens transformers einops datasets jaxtyping fancy_einsum -i "$PIP_INDEX" -q

# 6. 验证
python -c "
import torch; print(f\"torch {torch.__version__}, CUDA {torch.cuda.is_available()}\")
import transformer_lens; print(\"transformer_lens OK\")
import sae_lens; print(f\"sae_lens {sae_lens.__version__}\")
print(\"All checks passed!\")
"
