set -x
# ray start --head --node-ip-address 0.0.0.0 --num-gpus 4
#  "HF_ENDPOINT": "https://hf-mirror.com"

   # --vllm_num_engines 2 \
   # --vllm_tensor_parallel_size 2 \
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": ".",
                        "env_vars": {"CUDA_VISIBLE_DEVICES": "1,2"} }' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 1 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 1 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 1 \
   --colocate_critic_reward \
   --colocate_actor_ref \
   --ref_reward_offload \
   --pretrain meta-llama/Llama-3.2-1B-Instruct \
   --reward_pretrain Ray2333/Gemma-2B-rewardmodel-ft \
   --save_path /data/fist_user/checkpoint/Llama-3.2-1B-rlhf \
   --micro_train_batch_size 8 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 12 \
   --rollout_batch_size 1024 \
   --max_samples 100000 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --generate_max_len 512 \
   --zero_stage 2 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data OpenRLHF/prompt-collection-v0.1 \
   --input_key context_messages \
   --apply_chat_template \
   --normalize_reward \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing \
   --load_checkpoint \
   --use_wandb 054d76b9854601b58c2e36deb9c055965cef7cdf

# --vllm_num_engines 1 \
# --vllm_tensor_parallel_size 1 \
# --ckpt_path /data/fist_user/checkpoint/Llama-3.2-1B-rlhf \
# --runtime-env-json='{"setup_commands": ["pip install openrlhf[vllm]"]}' [Install deps]
# --ref_reward_offload [Offload to CPU]

# --save_steps 250 \
# --ckpt_path /data/fist_user/checkpoint/Llama-3.2-1B-rlhf \