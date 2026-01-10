# VRAG 训练流程总结

## 1. 整体架构概览

```
train_grpo_qwen2_5_vl_7b.sh
    │
    ▼
main_ppo_vrag.py (入口)
    │
    ├── Ray 初始化
    ├── TaskRunner.run() [远程Actor]
    │       │
    │       ├── 加载 tokenizer/processor
    │       ├── 创建 Worker 类映射
    │       ├── 创建 RMManager (奖励管理器)
    │       └── 创建 RayPPOTrainerVRAG
    │
    ▼
RayPPOTrainerVRAG.fit() (训练主循环)
    │
    ├── 数据加载 → DataProto
    ├── LLMGenerationManager.run_llm_loop() (多轮Agent交互)
    ├── 计算 log_prob (Actor/Ref)
    ├── RMManager.__call__() (奖励计算)
    ├── compute_advantage() (优势估计)
    └── update_actor/critic (策略更新)
```

## 2. 文件位置索引

| 组件 | 文件路径 |
|------|----------|
| 训练脚本 | `code/examples/train_grpo_qwen2_5_vl_7b.sh` |
| 入口文件 | `code/verl/verl/trainer/main_ppo_vrag.py` |
| 训练器 | `code/verl/verl/trainer/ppo/ray_trainer_vrag.py` |
| Agent生成 | `code/vrag_agent/generation.py` |
| 奖励管理 | `code/verl/verl/workers/reward_manager/rm.py` |
| 配置文件 | `code/verl/verl/trainer/config/ppo_trainer.yaml` |
| 搜索引擎（可供agent调用的工具） | `code/search_engine/search_engine_api.py` |


---

## 3. 训练脚本启动流程 (`train_grpo_qwen2_5_vl_7b.sh`)

```bash
# 关键配置
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
export VLLM_USE_V1=1

# 启动训练
python3 main_ppo_vrag.py \
    algorithm.adv_estimator=grpo \           # 使用GRPO优势估计
    data.train_files=vidoseek_train.parquet  # 训练数据
    actor_rollout_ref.model.path=Qwen2.5-VL-7B-Instruct  # 模型路径
```

---

## 4. main_ppo_vrag.py 入口流程

**文件**: `code/verl/verl/trainer/main_ppo_vrag.py`

```python
@hydra.main(config_path='config', config_name='ppo_trainer')
def main(config):
    # 1. Ray初始化
    ray.init(runtime_env={...})
    
    # 2. 创建远程TaskRunner
    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))

class TaskRunner:
    def run(self, config):
        # 3. 加载tokenizer和processor
        tokenizer = hf_tokenizer(local_path)
        processor = hf_processor(local_path)
        
        # 4. 创建Worker类映射
        role_worker_mapping = {
            Role.ActorRollout: ActorRolloutRefWorker,
            Role.Critic: CriticWorker,
            Role.RefPolicy: ActorRolloutRefWorker
        }
        
        # 5. 创建奖励管理器
        reward_fn = RMManager(tokenizer, use_local_rm=True, ...)
        
        # 6. 创建训练器并启动
        trainer = RayPPOTrainerVRAG(config, tokenizer, processor, ...)
        trainer.init_workers()
        trainer.fit()  # ← 进入主训练循环
```


---

## 5. RayPPOTrainerVRAG.fit() 详细流程

**文件**: `code/verl/verl/trainer/ppo/ray_trainer_vrag.py`
**函数**: `fit()` (第700-900行)

```
fit()
│
├── [初始化阶段]
│   ├── 创建 Tracking logger (wandb/console)
│   ├── _load_checkpoint() - 加载检查点
│   └── 创建 LLMGenerationManager
│
├── [训练循环] for epoch in total_epochs:
│   │
│   └── for batch_dict in train_dataloader:
│       │
│       ├── Step 数据准备
│       │   ├── DataProto.from_single_dict(batch_dict)
│       │   └── batch.repeat_deepcopy(n_agent=2) # 每样本生成2个候选
│       │
│       ├── Step 2: 多轮Agent生成 [gen]
│       │   └── generation_manager.run_llm_loop()
│       │       ├── 模型生成响应
│       │       ├── 解析动作 (search/answer/bbox)
│       │       ├── 执行环境交互
│       │       └── 更新上下文，循环直到结束
│       │
│       ├── Step 3: 计算 old_log_prob [old_log_prob]
│       │   └── actor_rollout_wg.compute_log_prob(batch)
│       │
│       ├── Step 4: 计算 ref_log_prob [ref] (可选)
│       │   └── ref_policy_wg.compute_ref_log_prob(batch)
│       │
│       ├── Step 5: 计算奖励 [adv]
│       │   ├── reward_fn(batch) → RMManager.__call__()
│       │   │   ├── 格式评分 (compute_score)
│       │   │   ├── 模型评分 (LocalRewardModel)
│       │   │   └── NDCG检索质量评分
│       │   │   └── 最终: 0.7*模型 + 0.1*格式 + 0.2*NDCG
│       │   │
│       │   ├── apply_kl_penalty() - KL惩罚
│       │   └── compute_advantage() - GRPO优势估计
│       │
│       ├── Step 6: 更新Critic [update_critic]
│       │   └── critic_wg.update_critic(batch)
│       │
│       ├── Step 7: 更新Actor [update_actor]
│       │   ├── _create_loss_mask() - 创建损失掩码
│       │   └── actor_rollout_wg.update_actor(batch)
│       │
│       ├── Step 8: 保存检查点 [save_checkpoint]
│       │   └── _save_checkpoint() (每save_freq步)
│       │
│       └── Step 9: 验证 [testing]
│           └── _validate() (每test_freq步)
│
└── [结束] 返回最终验证指标
```


---

## 6. LLMGenerationManager.run_llm_loop() 详细流程

**文件**: `code/vrag_agent/generation.py`
**函数**: `run_llm_loop()` (第450-620行)

```
run_llm_loop(gen_batch, initial_input_ids)
│
├── [初始化]
│   ├── original_left_side = {'input_ids': initial_input_ids}  # 原始prompt
│   ├── original_right_side = {'responses': empty}              # 响应序列
│   ├── active_mask = ones(batch_size)                          # 活跃掩码
│   ├── rollings = gen_batch                                    # 滚动上下文
│   └── retrievaled_images = [[] for _ in batch_size]           # 检索图像列表
│
├── [主循环] for step in range(max_turns=4):
│   │
│   ├── Step 1: 检查是否全部结束
│   │   └── if not active_mask.sum(): break
│   │
│   ├── Step 2: 裁剪序列到有效长度
│   │   └── tensor_fn.cut_to_effective_len(rollings.batch)
│   │
│   ├── Step 3: 更新raw_prompt_ids
│   │   └── _raw_prompt_ids(rollings)
│   │
│   ├── Step 4: 检查序列长度，停用超长样本
│   │   └── deactivate_batch(active_mask, rollings)
│   │
│   ├── Step 5: 提取活跃样本
│   │   └── rollings_active = rollings[active_mask]
│   │
│   ├── Step 6: 模型生成响应
│   │   └── gen_output = _generate_with_gpu_padding(rollings_active)
│   │       └── actor_rollout_wg.generate_sequences()
│   │
│   ├── Step 7: 后处理响应
│   │   └── _postprocess_responses(gen_output.batch['responses'])
│   │       └── 提取 <think>, <search>, <answer>, <bbox> 标签内容
│   │
│   ├── Step 8: 执行环境交互
│   │   └── execute_predictions(responses_str, ...)
│   │       ├── postprocess_predictions() - 解析动作类型
│   │       └── 根据动作执行:
│   │           ├── 'answer' → done=1, 结束该样本
│   │           ├── 'search' → 调用搜索引擎API获取图像
│   │           ├── 'bbox'   → 裁剪图像区域
│   │           └── invalid  → 返回错误提示
│   │
│   ├── Step 9: 更新活跃掩码
│   │   └── active_mask = active_mask * ~dones
│   │
│   ├── Step 10: 处理环境观察
│   │   └── _process_next_obs(next_obs, rollings)
│   │       ├── 处理检索到的图像
│   │       ├── 处理裁剪后的图像
│   │       └── 生成下一轮输入token
│   │
│   ├── Step 11: 拼接多模态数据
│   │   └── _concat_multi_modal_data(rollings, ...)
│   │
│   └── Step 12: 更新状态
│       ├── _update_rolling_state() - 更新滚动上下文
│       └── _update_right_side()    - 更新响应序列
│
├── [最终生成] if active_mask.sum():
│   └── 对剩余活跃样本再做一次生成
│
├── [后处理]
│   ├── _add_noisy_multi_modal_data() - 添加噪声图像(训练时)
│   └── 保存 retrievaled_images 到 non_tensor_batch
│
└── [返回]
    └── _compose_final_output(left_side, right_side, meta_info, rollings)
```


---

## 7. 奖励计算流程 (RMMacode/verl/workers/reward_manager/rm.
**函数**: `__call__()` (第200-320行)

```
RMr.__call__(data: DataProto)
│
├── 备评估数据]
│   └── for i in range(len(解码 prompt 和 response
│    <answer>...<nswer> 中的答案
│       _eval 列表
│
├── [格式评分]
│   └── compute(datae, solution_str, gd_truth, extra_info)
│   ─ 检查响应格式是否正确
│
├── [模型评分] (仅)
│   └── local_batch_sco_be_eval)
│       └── LocalRewardModel.score(query, response, reference)
│           ├── 构建对话格式
│           ├── Skywork-Rewar推理
│           └── sigmoid(logits) → 0/1 二值化
│
├── [NDCG检索质量评分]
│   └── ndcg(retrievaled_images, reference_images)
│       ├── 计算 DCG (折扣累积增益)
│       └── 归一化得到 NDCG
│
└── [最终奖励]
    └── score = 0.7 * model_eval_score + 0.1 * format_score + 0.2 * ndcg_value
```

---

## 8. 关键数据结构

### DProto
```python
DataProto:
    batch: Dict[str, Tensor]
        - input_ids: [batch, seq_len]
        - attention_mask: [batch, seq_len]
        - position_ids: [batch, 4, seq_len]  # Qwen2-VL 3D位置编码
        - responses: [batch, response_len]
        - prompts: [batch, prompt_len]
        - old_log_probs: [batch, seq_len]
        - token_level_scores: [batch, seq_len]
        - advantages: [batch, seq_len]
        
    non_tensor_batch: Dict[str, np.ndarray]
        - raw_prompt_ids: List[List[int]]
        - raw_prompt: List[Dict]
        - multi_modal_data: List[Dict]  # {'image': [PIL.Image]}
        - multi_modal_inputs: List[BatchFeature]
        - extra_info: List[Dict]  # {'question', 'answer', 'file_name', 'reference_page'}
        - retrievaled_images: List[List[str]]  # 检索到的图像路径
        - data_source: List[str]
        - uid: List[str]
```

### GenerationConfig
```python
GenerationConfig:
    max_turns: int = 4           # 最大交互轮数
    max_prompt_length: int       # 最大prompt长度
    num_gpus: int                # GPU数量
    n_agent: int = 2             # 每样本生成候选数
    search_url: str              # 搜索引擎URL
    max_model_len: int = 10240   # 模型最大长度
```


---

## 9. Agent动作类型

| 动作 | 格式 | 说明 |
|------|------|------|
| `search` | `<search>query</search>` | 调用搜索引擎检索相关图像 |
| `answer` | `<answer>answer</answer>` | 给出最终答案，结束交互 |
| `bbox` | `<bbox>[x1,y1,x2,y2]</bbox>` | 裁剪图像指定区域 |
| `think` | `<think>reasoning</think>` | 推理过程（不触发动作） |

---

## 10. 训练超参数 (默认值)

```yaml
# 数据
train_batch_size: 8
max_prompt_length: 8192
max_response_length: 2048

# Actor
actor.optim.lr: 1e-7
actor.ppo_mini_batch_size: 4 * n_gpus
actor.use_kl_loss: True
actor.kl_loss_coef: 0.01

# Rollout (vLLM)
rollout.temperature: 0.7
rollout.top_p: 0.9
rollout.gpu_memory_utilization: 0.5
rollout.n_agent: 2

# 训练
total_epochs: 3
save_freq: 25
test_freq: 25
max_turns: 4

# 奖励权重
model_score_weight: 0.7
format_score_weight: 0.1
ndcg_weight: 0.2
```

---

## 11. 完整调用链路图

```
[Shell Script]
train_grpo_qwen2_5_vl_7b.sh
    │
    ▼
[Python Entry]
main_ppo_vrag.py::main()
    │
    ├── ray.init()
    └── TaskRunner.run()
            │
            ├── hf_tokenizer() / hf_processor()
            ├── RMManager()
            └── RayPPOTrainerVRAG()
                    │
                    ├── __init__()
                    │   ├── _validate_config()
                    │   └── _create_dataloader()
                    │
                    ├── init_workers()
                    │   ├── ResourcePoolManager.create_resource_pool()
                    │   ├── actor_rollout_wg.init_model()
                    │   ├── critic_wg.init_model()
                    │   └── ref_policy_wg.init_model()
                    │
                    └── fit()  ← 主训练循环
                            │
                            ├── LLMGenerationManager.run_llm_loop()
                            │       │
                            │       ├── _generate_with_gpu_padding()
                            │       │       └── actor_rollout_wg.generate_sequences()
                            │       │
                            │       ├── _postprocess_responses()
                            │       ├── execute_predictions()
                            │       │       └── requests.get(search_url)
                            │       │
                            │       ├── _process_next_obs()
                            │       └── _update_rolling_state()
                            │
                            ├── actor_rollout_wg.compute_log_prob()
                            ├── ref_policy_wg.compute_ref_log_prob()
                            │
                            ├── RMManager.__call__()
                            │       ├── compute_score()
                            │       ├── LocalRewardModel.batch_score()
                            │       └── ndcg()
                            │
                            ├── apply_kl_penalty()
                            ├── compute_advantage()
                            │
                            ├── critic_wg.update_critic()
                            └── actor_rollout_wg.update_actor()
```
