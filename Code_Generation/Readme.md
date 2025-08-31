# The codebase for *RewardDS* on Code Generation Task

> Step 1. On the Client Side, DP finetune the Generation Proxy Model and Reward Proxy Model

```bash
sh prepare_client.sh
```

> Step 2. On the Server Side, apply **Reward Guided Filtering** and **Self-Optimizing Refinement** during LLM fine-tuning  

```bash
sh run_server.sh
```

> Step 2. Evaluate the performance of the fine-tuned LLM

```bash
sh evaluate.sh
```