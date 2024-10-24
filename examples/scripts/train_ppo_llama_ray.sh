set -x 

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/home/jovyan/notebook/zhouyang/OpenRLHF"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 2 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 2 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 1 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 1 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 1 \
   --colocate_critic_reward \
   --colocate_actor_ref \
   --pretrain /pubshare/zy/cache/Llama-3-8b-sft-mixture \
   --reward_pretrain /pubshare/zy/cache/Llama-3-8b-rm-mixture \
   --save_path /pubshare/zy/cache/checkpoint/llama-3-8b-rlhf \
   --micro_train_batch_size 1 \
   --train_batch_size 1 \
   --micro_rollout_batch_size 1 \
   --rollout_batch_size 1 \
   --max_samples 100000 \
   --max_epochs 1 \
   --prompt_max_len 512 \
   --generate_max_len 512 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data /home/jovyan/notebook/zhouyang/OpenRLHF/1data1PPO/train.json \
   --input_key prompt \
   --apply_chat_template \
   --normalize_reward \
   --flash_attn \
   --gradient_checkpointing \
   --load_checkpoint \
   --use_wandb {wandb_token}
#   --adam_offload \
#   --colocate_critic_reward \
#   --colocate_actor_ref \
# --runtime-env-json='{"setup_commands": ["pip install openrlhf[vllm]"]}' [Install deps]
# --ref_reward_offload [Offload to CPU]
