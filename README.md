verl安装：https://verl.readthedocs.io/en/v0.5.x/start/install.html

demo:
pip install -r requirements_demo.txt

python search_engine/search_engine_api.py
vllm serve ./models/Qwen3-VL-8B-Instruct --port 8001 --host 0.0.0.0 --limit-mm-per-prompt image=10 --served-model-name Qwen/Qwen3-VL-8B-Instruct

python search_engine/search_engine_api.py
vllm serve ./models/Qwen2.5-VL-7B-Instruct --port 8001 --host 0.0.0.0 --limit-mm-per-prompt image=10 --served-model-name Qwen/Qwen2.5-VL-7B-Instruct

streamlit run demo/app.py

