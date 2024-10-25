set -x 
wandb_token=b0255391060d68833e9b98941b9eb94fe770fbe4
deepspeed --master_port 29601 --include=localhost:1,2 --module openrlhf.cli.train_ppo_with_prm \
  --pretrain /home/jovyan/share/LLMAgent/model/Llama-3.2-1B-Instruct \
  --reward_pretrain /pubshare/LLM/math-shepherd-mistral-7b-prm \
  --critic_pretrain /pubshare/LLM/math-shepherd-mistral-7b-prm \
  --save_path /pubshare/fwk/orlhf_checkpoints/checkpoint/llama-32—1B-rlhf \
  --save_steps -1 \
  --logging_steps 1 \
  --eval_steps -1 \
  --micro_train_batch_size 1 \
  --train_batch_size 4 \
  --micro_rollout_batch_size 2 \
  --rollout_batch_size 8 \
  --max_epochs 1 \
  --prompt_max_len 512 \
  --generate_max_len 512 \
  --zero_stage 2 \
  --bf16 \
  --actor_learning_rate 5e-7 \
  --critic_learning_rate 9e-6 \
  --init_kl_coef 0.01 \
  --prompt_data /home/jovyan/notebook/fwk/OpenRLHF/dataset/math/train.jsonl \
  --input_key problem \
  --max_samples 100000 \
  --normalize_reward \
  --adam_offload \
  --flash_attn \
  --gradient_checkpointing \
  --use_wandb $wandb_token