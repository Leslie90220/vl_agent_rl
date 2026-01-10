python - <<'PY'
import re, sys, subprocess
from importlib import metadata

reqs = r"""
accelerate
codetiming
datasets
dill
flash-attn
hydra-core
liger-kernel
math-verify[antlr4_9_3]
numpy
pandas
peft
pyarrow>=15.0.0
pybind11
pylatexenc
ray[default]
tensordict<0.6
torchdata
transformers
vllm==0.8.2
wandb
openai
streamlit
git+https://github.com/illuin-tech/colpali@481eb200834f32b87f66dec34c0deb7cd4434146
transformers>=4.50.3
llama-cloud==0.1.5
llama-index==0.12.0
llama-index-agent-openai==0.4.0
llama-index-cli==0.4.0
llama-index-core==0.12.1
llama-index-embeddings-huggingface==0.4.0
llama-index-embeddings-openai==0.3.0
llama-index-indices-managed-llama-cloud==0.6.2
llama-index-legacy==0.9.48.post4
llama-index-llms-openai==0.3.1
llama-index-multi-modal-llms-openai==0.3.0
llama-index-program-openai==0.3.0
llama-index-question-gen-openai==0.3.0
llama-index-readers-file==0.4.0
llama-index-readers-llama-parse==0.4.0
llama-parse==0.5.14
""".strip().splitlines()

def norm_name(s: str) -> str:
    # 取出包名（去掉 extras / 版本约束 / git+...）
    s = s.strip()
    if not s or s.startswith("#"): 
        return ""
    if s.startswith("git+"):
        # 你这个是 colpali 的源码安装，通常 distribution 名叫 colpali_engine
        # 但也可能叫 colpali；这里两种都检查
        return "colpali_engine|colpali"
    s = re.split(r"[<>=!~\[]", s, 1)[0]
    return s

def get_version(dist_name: str):
    try:
        return metadata.version(dist_name)
    except Exception:
        return None

print("Python:", sys.version.split()[0])
try:
    import torch
    print("Torch :", torch.__version__, " CUDA:", torch.version.cuda)
except Exception as e:
    print("Torch import failed:", e)

print("\n=== Package presence/version ===")
missing = []
for raw in reqs:
    name = norm_name(raw)
    if not name:
        continue
    if "|" in name:  # special case for git colpali
        candidates = name.split("|")
        found = False
        for c in candidates:
            v = get_version(c)
            if v:
                print(f"{raw:70s} -> INSTALLED  {c}=={v}")
                found = True
                break
        if not found:
            print(f"{raw:70s} -> MISSING")
            missing.append(raw)
        continue

    v = get_version(name.replace("_","-")) or get_version(name)
    if v:
        print(f"{raw:70s} -> INSTALLED  {name}=={v}")
    else:
        print(f"{raw:70s} -> MISSING")
        missing.append(raw)

print("\n=== Missing ===")
for m in missing:
    print(m)
PY
