import logging
import time
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import re

import ray
import torch
import torch.nn as nn
from tqdm import tqdm

from openrlhf.models.actor import Actor
from openrlhf.models.utils import compute_reward, masked_mean, unpacking_samples, masked_sum
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.remote_rm_utils import remote_rm_fn, remote_rm_fn_ray

logger = init_logger(__name__)


def to(tensor: Union[torch.Tensor, list[torch.Tensor]], device):
    if isinstance(tensor, list):
        return [to(t, device) for t in tensor]
    return tensor.to(device)


def pin_memory(tensor: Union[torch.Tensor, list[torch.Tensor]]):
    if isinstance(tensor, list):
        return [pin_memory(t) for t in tensor]
    return tensor.pin_memory()


@dataclass
class Experience:
    """Experience is a batch of data.
    These data should have the the sequence length and number of actions.
    Left padding for sequences is applied.

    Shapes of each tensor:
    sequences: (B, S)
    action_log_probs: (B, A)
    values: (B, A)
    returns: (B, A)
    advatanges: (B, A)
    attention_mask: (B, S)
    action_mask: (B, A)

    "A" is the number of actions.
    """

    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    info: Optional[dict]

    @torch.no_grad()
    def to_device(self, device: torch.device) -> None:
        self.sequences = to(self.sequences, device)
        self.action_log_probs = to(self.action_log_probs, device)
        self.values = to(self.values, device)
        self.returns = to(self.returns, device)
        self.advantages = to(self.advantages, device)
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.to(device)
        if self.action_mask is not None:
            self.action_mask = self.action_mask.to(device)

    def pin_memory(self):
        self.sequences = pin_memory(self.sequences)
        self.action_log_probs = pin_memory(self.action_log_probs)
        self.values = pin_memory(self.values)
        self.returns = pin_memory(self.returns)
        self.advantages = pin_memory(self.advantages)
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.pin_memory()
        if self.action_mask is not None:
            self.action_mask = self.action_mask.pin_memory()
        return self


class NaiveExperienceMaker(ABC):
    """
    Naive experience maker.
    """

    def __init__(
        self,
        actor: Actor,
        critic: nn.Module,
        reward_model: nn.Module,
        initial_model: Actor,
        tokenizer,
        critic_tokenizer,
        prompt_max_len: int,
        kl_controller,
        strategy=None,
        remote_rm_url: str = None,
        reward_fn=None,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.initial_model = initial_model
        self.tokenizer = tokenizer
        self.critic_tokenizer = critic_tokenizer
        self.prompt_max_len = prompt_max_len
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.reward_fn = reward_fn

    # tokenizer
    def tokenize_fn(self, texts, max_length, padding=True, device=None, tokenizer=None):
        if tokenizer is None:
            tokenizer = self.tokenizer
        if not padding:
            # when padding is False, return tokenized texts as list
            return tokenizer(
                texts,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
        batch = tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    @torch.no_grad()
    def make_experience(self, prompts: Union[str, List[str]], **generate_kwargs) -> Experience:
        self.actor.eval()
        self.critic.eval()
        self.initial_model.eval()
        if self.reward_model is not None:
            self.reward_model.eval()

        # generate seq
        inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
        
        self.strategy.print('actor use gpu:'+str(next(self.actor.parameters()).is_cuda))
        self.strategy.print('='*30+'actor start to generate sequences'+30*'=')
        sequences, attention_mask, action_mask = self.actor.generate(**inputs, **generate_kwargs)
        self.strategy.print('='*30+'actor finish generating sequences'+30*'=')
        num_actions = action_mask.size(1)


        # log probs
        self.strategy.print('='*30+'actor start to get log probs'+30*'=')
        action_log_probs = self.actor(sequences, num_actions, attention_mask)
        self.strategy.print('='*30+'actor finish getting log probs'+30*'=')

        # init log probs
        self.strategy.print('initial model use gpu:'+str(next(self.initial_model.parameters()).is_cuda))
        # for parameter in self.initial_model.parameters():
        #     if not parameter.is_cuda:
        #         self.strategy.print('initial model parameter not on gpu')
        #         breakpoint()
        self.strategy.print('='*30+'initial model start to get log probs'+30*'=')
        base_action_log_probs = self.initial_model(sequences, num_actions, attention_mask)
        self.strategy.print('='*30+'initial model finish getting log probs'+30*'=')

        # values
        self.strategy.print('critic use gpu:'+str(next(self.critic.parameters()).is_cuda))
        self.strategy.print('='*30+'critic start to get values'+30*'=')
        value = self.critic(sequences, num_actions, attention_mask)
        self.strategy.print('='*30+'critic finish getting values'+30*'=')

        # rewards
        self.strategy.print('reward model use gpu:'+str(next(self.reward_model.parameters()).is_cuda))
        self.strategy.print('='*30+'reward model start to get rewards'+30*'=')
        if self.remote_rm_url is not None:
            # remote RM
            queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=False)
            r = remote_rm_fn(self.remote_rm_url, queries=queries).to(device=action_log_probs.device)
        else:
            # local RM
            r = self.reward_model(sequences, attention_mask)
        self.strategy.print('='*30+'reward model finish getting rewards'+30*'=')

        self.strategy.print('='*30+'compute reward'+30*'=')
        reward, kl = compute_reward(
            r,
            self.kl_ctl.value,
            action_log_probs,
            base_action_log_probs,
            action_mask=action_mask,
        )
        # self.strategy.print('='*30+'finish compute reward'+30*'=')

        # self.strategy.print('='*30+'get advantages and returns'+30*'=')
        advantage, returns = self.get_advantages_and_returns(
            value,
            reward,
            action_mask,
            generate_kwargs["gamma"],
            generate_kwargs["lambd"],
        )
        self.strategy.print('='*30+'finish get advantages and returns'+30*'=')

        info = {
            "kl": masked_mean(kl, action_mask, dim=-1),
            "reward": r,
            "return": reward.sum(dim=-1),
            "response_length": action_mask.float().sum(dim=-1),
            "total_length": attention_mask.float().sum(dim=-1),
        }
        # reset model state
        self.actor.train()
        self.critic.train()

        return Experience(
            sequences,
            action_log_probs,
            value,
            returns,
            advantage,
            attention_mask,
            action_mask,
            info,
        )

    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)

        Output:
        - advantages: Tensor of shape (batch_size, response_size)
        - returns: Tensor of shape (batch_size, response_size)
        """
        if isinstance(values, list):
            # packing samples
            # TODO: this is slow...
            advantages = []
            returns = []
            for v, r in zip(values, rewards):
                adv, ret = self.get_advantages_and_returns(v.unsqueeze(0), r.unsqueeze(0), action_mask, gamma, lambd)
                advantages.append(adv.squeeze(0))
                returns.append(ret.squeeze(0))
            return advantages, returns

        # gae, generalized advantage estimation 
        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        if action_mask is not None:
            values = action_mask * values
            rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            # lambd 是 GAE 中的平滑系数，控制优势估计的偏差与方差平衡
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns


class RemoteExperienceMaker(NaiveExperienceMaker):
    def __init__(self, *args, vllm_engines: List = None, packing_samples=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.vllm_engines = vllm_engines
        self.packing_samples = packing_samples

    @torch.no_grad()
    def make_experience(self, prompts: Union[str, List[str]], **generate_kwargs) -> Experience:
        self.actor.eval()
        device = torch.cuda.current_device()

        # generate sequence
        start = time.time()
        if not self.packing_samples:
            sequences, attention_mask, action_mask = (
                self._generate_local(prompts, **generate_kwargs)
                if self.vllm_engines is None
                else self._generate_vllm(prompts, **generate_kwargs)
            )
            num_actions = action_mask.size(1)
            packed_seq_lens = None
            response_length = action_mask.float().sum(dim=-1)
            total_length = attention_mask.float().sum(dim=-1)
        else:
            assert self.vllm_engines is not None, "vllm_engines must be provided for packed samples"
            sequences, attention_mask, packed_seq_lens, num_actions = self._generate_vllm(prompts, **generate_kwargs)
            action_mask = None
            response_length = torch.tensor(num_actions, device=device, dtype=torch.float)
            total_length = torch.tensor(packed_seq_lens, device=device, dtype=torch.float)
        generate_time = time.time() - start

        sequences_cpu, attention_mask_cpu = (
            sequences.to("cpu"),
            attention_mask.to("cpu"),
        )

        # init log probs
        base_action_log_probs_ref = self.initial_model.forward.remote(
            sequences_cpu, num_actions, attention_mask_cpu, packed_seq_lens=packed_seq_lens
        )

        # values
        value_ref = self.critic.forward.remote(
            sequences_cpu, num_actions, attention_mask_cpu, packed_seq_lens=packed_seq_lens
        )

        # avoid CUDA OOM when colocate models
        if self.strategy.args.colocate_critic_reward:
            ray.get([value_ref])
            ray.get([self.critic.empty_cache.remote()])

        if self.strategy.args.colocate_actor_ref:
            ray.get([base_action_log_probs_ref])
            ray.get([self.initial_model.empty_cache.remote()])

        # rewards
        r_refs = []
        # support remote RM API with ray
        if not self.remote_rm_url:
            for rm in self.reward_model:
                r_refs.append(rm.forward.remote(sequences_cpu, attention_mask_cpu, packed_seq_lens=packed_seq_lens))
        else:
            # remote RM
            for rm in self.remote_rm_url:
                if not self.packing_samples:
                    queries = self.tokenizer.batch_decode(sequences_cpu, skip_special_tokens=False)
                    r = remote_rm_fn_ray.remote(rm, queries=queries)
                    r_refs.append(r)
                else:
                    sequences_list = []
                    offset = 0
                    tokens_list = sequences_cpu.tolist()[0]
                    for length in packed_seq_lens:
                        sequences_list.append(tokens_list[offset : offset + length])
                        offset += length
                    queries = self.tokenizer.batch_decode(sequences_list, skip_special_tokens=False)
                    r = remote_rm_fn_ray.remote(rm, queries=queries)
                    r_refs.append(r)

        # log probs
        start = time.time()
        action_log_probs = self.actor(sequences, num_actions, attention_mask, packed_seq_lens=packed_seq_lens)
        actor_time = time.time() - start

        # wait initial/critic/reward model done
        start = time.time()
        ref_values = ray.get([base_action_log_probs_ref, value_ref] + r_refs)
        wait_time = time.time() - start

        base_action_log_probs, value, rewards = ref_values[0], ref_values[1], ref_values[2:]
        base_action_log_probs, value = base_action_log_probs.to(device), value.to(device)
        rewards = [r.to(device) for r in rewards]
        r = self.reward_fn(rewards) if len(rewards) > 0 else rewards[0]

        # avoid CUDA OOM when colocate models
        if self.strategy.args.colocate_critic_reward and not self.remote_rm_url:
            ray.get([self.reward_model[0].empty_cache.remote()])

        if self.strategy.args.colocate_actor_ref:
            torch.cuda.empty_cache()

        reward, kl = compute_reward(
            r,
            self.kl_ctl.value,
            action_log_probs,
            base_action_log_probs,
            action_mask=action_mask,
            num_actions=num_actions,
        )

        if not self.packing_samples:
            kl = masked_mean(kl, action_mask, dim=-1)
            return_sums = reward.sum(dim=-1)
        else:
            # convert tensor into list of tensors so that it's easier to manipulate
            # within dataset.
            sequences = unpacking_samples(sequences, packed_seq_lens)
            attention_mask = None
            action_log_probs = unpacking_samples(action_log_probs, num_actions)
            value = unpacking_samples(value, num_actions)
            reward = unpacking_samples(reward, num_actions)

            kl = unpacking_samples(kl, num_actions)
            kl = torch.tensor([each_kl.mean() for each_kl in kl], device=device)
            return_sums = torch.tensor([each_reward.sum() for each_reward in reward], device=device)

        advantage, returns = self.get_advantages_and_returns(
            value,
            reward,
            action_mask,
            generate_kwargs["gamma"],
            generate_kwargs["lambd"],
        )

        info = {
            "kl": kl,
            "reward": r,
            "return": return_sums,
            "response_length": response_length,
            "total_length": total_length,
        }

        if self.strategy.args.perf:
            batch_size = 1 if isinstance(prompts, str) else len(prompts)
            info["generate_time"] = torch.full((batch_size,), generate_time, device=device)
            info["actor_time"] = torch.full((batch_size,), actor_time, device=device)
            info["wait_time"] = torch.full((batch_size,), wait_time, device=device)

        experience = Experience(
            sequences,
            action_log_probs,
            value,
            returns,
            advantage,
            attention_mask,
            action_mask,
            info,
        )

        # send experience to critic
        experience_cpu = deepcopy(experience)
        experience_cpu.to_device("cpu")
        self._ref = self.critic.append.remote(experience_cpu)

        self.actor.train()  # reset model state
        return experience

    def _generate_local(self, prompts: List[str], **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
        return self.actor.generate(**inputs, **kwargs)

    def _generate_vllm(self, prompts: List[str], **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from vllm import SamplingParams

        # round-robin load balance
        rank = torch.distributed.get_rank()
        llm = self.vllm_engines[rank % len(self.vllm_engines)]

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
        )

        # TODO: can't pass `max_length` to vLLM's tokenizer for input truncation, remove this once it is supported.
        prompt_token_ids = self.tokenize_fn(prompts, self.prompt_max_len, padding=False)["input_ids"]
        outputs = ray.get(llm.generate.remote(sampling_params=sampling_params, prompt_token_ids=prompt_token_ids))

        if not self.packing_samples:
            # NOTE: concat all outputs to following format:
            #
            # | [PAD] [PAD] token token token | token token [EOS] [PAD] |
            # | token token token token token | token token [EOS] [PAD] |
            # | [PAD] [PAD] [PAD] token token | token token token [EOS] |
            # |<---------- prompt ----------->|<-------- answer ------->|
            max_input_len, max_output_len = 0, 0
            for output in outputs:
                max_input_len = max(max_input_len, len(output.prompt_token_ids))
                max_output_len = max(max_output_len, len(output.outputs[0].token_ids))

            pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
            sequences = []
            for output in outputs:
                # left padding input
                input_len = len(output.prompt_token_ids)
                input_ids = [pad_token_id] * (max_input_len - input_len) + list(output.prompt_token_ids)

                # right padding output
                output_len = len(output.outputs[0].token_ids)
                output_ids = list(output.outputs[0].token_ids) + [pad_token_id] * (max_output_len - output_len)

                if output_ids[output_len - 1] != eos_token_id:
                    output_ids[min(output_len, len(output_ids) - 1)] = eos_token_id

                # concat input and output
                sequences.append(input_ids + output_ids)

            sequences = torch.tensor(sequences)
            sequences, attention_mask, action_mask = self.actor.process_sequences(
                sequences, max_input_len, eos_token_id, pad_token_id
            )
            return sequences.to("cuda"), attention_mask.to("cuda"), action_mask.to("cuda")
        else:
            # NOTE: concat all outputs to following format:
            #
            # | token token token | token token [EOS] | token token token token token | token token [EOS] | token token | token token token [EOS] |
            # |<---  prompt ----->|<---- answer ----->|<---------- prompt ----------->|<----- answer ---->|<- prompt -->|<-------- answer ------->|
            pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
            sequences = []
            packed_seq_lens = []
            attention_mask = []
            num_actions = []
            for i, output in enumerate(outputs):
                input_len = len(output.prompt_token_ids)
                output_len = len(output.outputs[0].token_ids)
                packed_seq_lens.append(input_len + output_len)
                sequences.extend(output.prompt_token_ids + list(output.outputs[0].token_ids))
                attention_mask.extend([i + 1] * (input_len + output_len))

                # current_action_mask = [0] * (input_len - 1) + [1] * output_len + [0]
                # num_actions.append(max(1, sum(current_action_mask)))
                num_actions.append(max(1, output_len))

            sequences = torch.tensor(sequences, device="cuda").unsqueeze(0)
            attention_mask = torch.tensor(attention_mask, device="cuda").unsqueeze(0)
            return sequences, attention_mask, packed_seq_lens, num_actions

    def flush(self):
        "Ensure all experience has been send to critic"
        ray.get(self._ref)
        self._ref = None

class PRMExperienceMaker(NaiveExperienceMaker):
    @torch.no_grad()
    def make_experience(self, prompts: Union[str, List[str]], **generate_kwargs) -> Experience:
        self.actor.eval()
        self.critic.eval()
        self.initial_model.eval()
        if self.reward_model is not None:
            self.reward_model.eval()

        # generate seq
        inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
        input_len = inputs['input_ids'].size(1)
        self.strategy.print('actor use gpu:'+str(next(self.actor.parameters()).is_cuda))
        self.strategy.print('='*30+'actor start to generate sequences'+30*'=')
        # sequences = prompt+answer
        sequences, attention_mask, action_mask = self.actor.generate(**inputs, **generate_kwargs)

        # sequences = sequences[...,:-5]
        # attention_mask = attention_mask[...,:-5]
        # action_mask = action_mask[...,:-5]
        # last_elements = sequences[:, -1]
        # seq_len = sequences.size(1)
        # expanded_sequences = torch.cat([sequences, last_elements.unsqueeze(1).expand(sequences.size(0), self.prompt_max_len+input_len - seq_len)], dim=1)
        # expanded_attention_mask = torch.cat([attention_mask, torch.ones(attention_mask.size(0), self.prompt_max_len - seq_len+input_len, device=attention_mask.device)], dim=1)
        # expanded_action_mask = torch.cat([action_mask, torch.ones(action_mask.size(0), self.prompt_max_len - seq_len+input_len, device=action_mask.device)], dim=1)
        # expanded_sequences[...,-1] = 128009
        # expanded_attention_mask[...,-1] = 1
        # expanded_action_mask[...,-1] = 1
        # sequences = expanded_sequences
        # attention_mask = expanded_attention_mask
        # action_mask = expanded_action_mask
        # breakpoint()

        self.strategy.print('='*30+'actor finish generating sequences'+30*'=')
        num_actions = action_mask.size(1)


        # log probs
        self.strategy.print('='*30+'actor start to get log probs'+30*'=')
        # 这取出来的只有answer部分(num_actions长度)
        action_log_probs = self.actor(sequences, num_actions, attention_mask)
        self.strategy.print('='*30+'actor finish getting log probs'+30*'=')

        # init log probs
        self.strategy.print('initial model use gpu:'+str(next(self.initial_model.parameters()).is_cuda))
        # for parameter in self.initial_model.parameters():
        #     if not parameter.is_cuda:
        #         self.strategy.print('initial model parameter not on gpu')
        #         breakpoint()
        self.strategy.print('='*30+'initial model start to get log probs'+30*'=')
        base_action_log_probs = self.initial_model(sequences, num_actions, attention_mask)
        self.strategy.print('='*30+'initial model finish getting log probs'+30*'=')

        # values
        self.strategy.print('critic use gpu:'+str(next(self.critic.parameters()).is_cuda))
        self.strategy.print('='*30+'critic start to get values'+30*'=')
        # value = self.critic(sequences, num_actions, attention_mask)
        # output = self.critic(sequences, attention_mask=attention_mask)
        value= self.compute_value_from_sequences(sequences, attention_mask, input_len=input_len, num_actions=num_actions)
        self.strategy.print('='*30+'critic finish getting values'+30*'=')

        # rewards
        self.strategy.print('reward model use gpu:'+str(next(self.reward_model.parameters()).is_cuda))
        self.strategy.print('='*30+'reward model start to get rewards'+30*'=')
        if self.remote_rm_url is not None:
            # remote RM
            queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=False)
            r = remote_rm_fn(self.remote_rm_url, queries=queries).to(device=action_log_probs.device)
        else:
            # local RM
            r, step_mask = self.compute_reward_from_output(sequences, attention_mask, input_len, num_actions=num_actions)
        self.strategy.print('='*30+'reward model finish getting rewards'+30*'=')
        self.strategy.print('='*30+'compute reward'+30*'=')
        # breakpoint()
        reward, kl = compute_reward(
            r,
            self.kl_ctl.value,
            action_log_probs,
            base_action_log_probs,
            action_mask=action_mask,
        )
        # self.strategy.print('='*30+'finish compute reward'+30*'=')

        # self.strategy.print('='*30+'get advantages and returns'+30*'=')
        advantage, returns = self.get_advantages_and_returns(
            value,
            reward,
            action_mask,
            generate_kwargs["gamma"],
            generate_kwargs["lambd"],
        )
        self.strategy.print('='*30+'finish get advantages and returns'+30*'=')

        info = {
            "kl": masked_mean(kl, action_mask, dim=-1),
            # "reward": r.mean(dim=-1),
            "reward": masked_mean(r, step_mask, dim=-1),
            # "token_reward": r,
            # "token_value": value,
            # "return": reward.sum(dim=-1),
            "return": masked_sum(reward, step_mask, dim=-1),
            "response_length": action_mask.float().sum(dim=-1),
            "total_length": attention_mask.float().sum(dim=-1),
        }
        # reset model state
        self.actor.train()
        self.critic.train()

        return Experience(
            sequences,
            action_log_probs,
            value,
            returns,
            advantage,
            attention_mask,
            action_mask,
            info,
        )
    
    def process_text(self, text: str)->str:
        text = text.replace('\r\n', '\n').replace('\n', 'ки')
        # text = re.sub(r'\n\s+\n', '\n', text)
        # km  km km 这种情况多换几次

        text = re.sub(r'ки\s+ки', 'ки', text)
        text = re.sub(r'ки\s+ки', 'ки', text)
        text = re.sub(r'ки\s+ки', 'ки', text)
        return text

    def find_first_matching_char(self, s: str)->int:
        # 使用集合提高查找效率
        allowed_chars = {'Ċ'}
        # 遍历字符串并检查每个字符
        for i, char in enumerate(s):
            if char in allowed_chars:
                return i
        return len(s)

    def compute_value_from_sequences(self, sequences: torch.Tensor, attention_mask: torch.Tensor, input_len: int, num_actions: int, return_output:bool=False):
        step_separater = '\n'
        km_token = 'ки'
        km_token_id1 = 1107
        km_token_id2 = 12902
        # 12902
        # km_token_id = self.critic_tokenizer.encode(km_token, add_special_tokens=False)[0]
        # +
        good_token_id = 648
        # good_token_id = self.critic_tokenizer.encode('+', add_special_tokens=False)[0]
        # -
        bad_token_id = 387
        # bad_token_id = self.critic_tokenizer.encode('-', add_special_tokens=False)[0]
        candidate_tokens = [good_token_id, bad_token_id]
        # \n 
        step_separater_id = 13
        # 198
        step_separater_id = 198
        # step_separater_id = self.tokenizer.encode(step_separater, add_special_tokens=False)[0]

        origin_seq_len = sequences.size(1)

        decoded_sequences = self.tokenizer.batch_decode(sequences, clean_up_tokenization_spaces=False)
        # decoded_sequences = [seq.replace(step_separater, km_token) for seq in decoded_sequences]
        # 处理ки   ки
        # decoded_sequences = [re.sub(r'ки\s+ки', 'ки ', seq) for seq in decoded_sequences]
        origin_decoded_sequences = decoded_sequences
        decoded_sequences = [self.process_text(seq) for seq in decoded_sequences]

        # reencoded_sequences = self.tokenizer.batch_encode_plus(
        #     decoded_sequences, 
        #     return_tensors='pt', 
        #     padding=False, 
        #     add_special_tokens=False
        # )
        reencoded_inputs = self.tokenize_fn(decoded_sequences, self.prompt_max_len*10, device='cuda', tokenizer=self.critic_tokenizer)
        reencoded_sequences = reencoded_inputs['input_ids']
        reencoded_attention_mask = reencoded_inputs['attention_mask']


        logits = self.critic(reencoded_sequences, attention_mask=reencoded_attention_mask, num_actions=-1)
        logits = logits[..., candidate_tokens]

        scores = logits.softmax(dim=-1)[:,:,0]

        # mask_1 = scores > 0.7
        # scores[mask_1] = 1
        # scores[~mask_1] = -1
        # scores = mask_1.float()*1.0 + (~mask_1).float()*(-1.0)
        scores = scores*2-1
        # scores.requires_grad = True
        # scores_1m1 = torch.where(scores > 0.7, torch.ones_like(scores, device=scores.device), torch.full_like(scores, -1.0, device=scores.device))
        
        values = torch.zeros_like(sequences, dtype=scores.dtype, device=scores.device)
        # tokens = self.tokenizer.convert_ids_to_tokens(reencoded_sequences[0].tolist())
        sep_mask = torch.zeros_like(sequences, dtype=torch.bool)
        # 批量转换 token IDs 为 tokens
        batch_tokens = [self.tokenizer.convert_ids_to_tokens(ids.tolist()) for ids in sequences]
        # 更新 mask，若 token 中包含 '\n' 或 '.Ċ' 则将对应位置的 mask 设为 True
        

        
        
        for i, tokens in enumerate(batch_tokens):
            # last_token = None
            # last_last_token = None
            tokens_pre = [None, None]
            for j, token in enumerate(tokens):
                # if token is None:
                #     with open('why_none.txt', 'w') as f:
                #         f.write(str(tokens)+'\n')
                #     breakpoint()
                if token is not None and (('\n' in token) or ('Ċ' in token)):
                    # if (last_token is None) or not sep_mask[i,j-1] or (last_token[-1]=='\n' or last_token[-1]=='Ċ') and (token[0]!='\n' or token[0]!='Ċ'):
                    last_token = tokens_pre[1]
                    last_last_token = tokens_pre[0]
                    if (last_token is None) or (not sep_mask[i,j-1]) or ((token[0]!='\n' or token[0]!='Ċ') and not (bool(re.match(r'^[ĠĊĉč]*$', token)))):
                        sep_mask[i, j] = True
                    if last_last_token is not None:
                        # last_last_token_reverse = last_last_token[::-1]
                        # 从右往左最后一个为空白的字符位置
                        # last_last_token_len = len(last_last_token_reverse)
                        last_last_token_first_non_matching_char_idx = self.find_first_matching_char(last_last_token)
                        # 从左往右最后一个为空白的字符位置
                        this_token_reverse = token[::-1]
                        this_token_reverse_first_non_matching_char_idx = self.find_first_matching_char(this_token_reverse)
                        slice_last_last_token = last_last_token[last_last_token_first_non_matching_char_idx+1:]
                        slice_this_token = this_token_reverse[this_token_reverse_first_non_matching_char_idx+1:][::-1]
                        if sep_mask[i, j-2] and bool(re.match(r'^[ĠĉčĊ]*$',slice_last_last_token+last_token+slice_this_token)):
                            sep_mask[i, j] = False
                tokens_pre = [tokens_pre[1], token]

        origin_sep_mask = sep_mask.clone()
        # sep_mask = sep_mask & ~torch.cat([torch.zeros(sep_mask.shape[0], 1, dtype=torch.bool, device=sep_mask.device), sep_mask[:, :-1]], dim=1)

        # sep_mask = sequences == step_separater_id
        km_mask = (reencoded_sequences == km_token_id1) | (reencoded_sequences == km_token_id2)
        # 连续的km 只保留第一个
        # km_mask = km_mask & ~km_mask.roll(1, 1)
        km_mask = km_mask & ~torch.cat([torch.zeros(km_mask.shape[0], 1, dtype=torch.bool, device=km_mask.device), km_mask[:, :-1]], dim=1)
        if km_mask.sum().item() != sep_mask.sum().item():
            with open('error.txt', 'w') as f:
                f.write('\n'+'='*100+'\n')
                f.write(f'km_mask: {km_mask.sum().item()}; sep_mask: {sep_mask.sum().item()}\n')
                f.write(f'km_mask shape: {km_mask.shape}; sep_mask shape: {sep_mask.shape}\n')
                f.write(f'origin km_mask: {((reencoded_sequences == km_token_id1) | (reencoded_sequences == km_token_id2)).sum().item()}; ')
                f.write(f'origin sep_mask: {origin_sep_mask.sum().item()}\n')
                for i, tokens in enumerate(batch_tokens):
                    for j in range(len(tokens)):
                        newline_token = ('\n' in tokens[j]) or ('Ċ' in tokens[j])
                        f.write(f'({i},{j}): {tokens[j]}; ac{newline_token}\n')
                    # f.write('\n'.join(tokens))
                    f.write('\n')
                # f.write(str(reencoded_sequences.tolist()))
                f.write('\n')
                critic_batch_tokens = [self.critic_tokenizer.convert_ids_to_tokens(ids.tolist()) for ids in reencoded_sequences]
                for i, tokens in enumerate(critic_batch_tokens):
                    for j in range(len(tokens)):
                        f.write(f'({i},{j}): {tokens[j]}; {reencoded_sequences[i][j]}\n')
                    # f.write('\n'.join(tokens))
                    f.write('\n')
                # f.write
                f.write(str(decoded_sequences)+'\n')
                f.write(str(origin_decoded_sequences))
            breakpoint()
        # values = values.clone()  # Create a clone if values need to maintain gradient tracking
        # values = torch.where(sep_mask, scores[km_mask].view(-1), values)
        # values[sep_mask] = scores[km_mask]
        
        selected_scores = torch.masked_select(scores, km_mask)

        # values_temp = values.clone() 
        
        values.masked_scatter_(sep_mask, selected_scores)
        # values = values_temp

        # values = scores.clone()
        # values = torch.tensor(scores, requires_grad=True)

        if values.size(1) > num_actions:
            values = values[:, -num_actions:]

        # scores_len = scores.size(1)
        # if values.size(1) < num_actions:
        #     # TODO: 全设为0名？
        #     new_values = torch.zeros(values.size(0), num_actions, device=values.device)
        #     new_values[:, :values.size(1)] = values
        #     values = new_values
        # else:
        # if scores_len < num_actions:
        #     values = torch.zeros(scores.size(0), num_actions)
        #     values[:, :scores_len] = scores
        # else:
        #     values = scores[:, -num_actions:]
        

        # compute_reward 中reward被限制在这个数量级
        # values = values.clamp(min=-10, max=10)
        if return_output:
            return values, None
        else:
            return values

    def compute_reward_from_output(self, sequences: torch.Tensor, attention_mask: torch.Tensor, input_len: int, num_actions: int):
        step_separater = '\n'
        km_token = 'ки'
        # 12902
        km_token_id1 = 1107
        km_token_id2 = 12902
        # km_token_id = self.critic_tokenizer.encode(km_token, add_special_tokens=False)[0]
        
        # +
        good_token_id = 648
        # good_token_id = self.critic_tokenizer.encode('+', add_special_tokens=False)[0]
        # -
        bad_token_id = 387
        # bad_token_id = self.critic_tokenizer.encode('-', add_special_tokens=False)[0]
        candidate_tokens = [good_token_id, bad_token_id]
        # \n 
        step_separater_id = 13
        step_separater_id = self.tokenizer.encode(step_separater, add_special_tokens=False)[0]

        decoded_sequences = self.tokenizer.batch_decode(sequences, clean_up_tokenization_spaces=False)
        # decoded_sequences = [seq.replace(step_separater, km_token) for seq in decoded_sequences]
        # 处理ки   ки
        # decoded_sequences = [re.sub(r'ки\s+ки', 'ки ', seq) for seq in decoded_sequences]
        decoded_sequences = [self.process_text(seq) for seq in decoded_sequences]

        reencoded_inputs = self.tokenize_fn(decoded_sequences, self.prompt_max_len*10, device='cuda', tokenizer=self.critic_tokenizer)

        reencoded_sequences = reencoded_inputs['input_ids']
        reencoded_attention_mask = reencoded_inputs['attention_mask']
        # eos_token_id, pad_token_id = self.tokenizer.eos_token_id, self.tokenizer.pad_token_id

        # attention_mask = (sequences.ne(eos_token_id) & sequences.ne(pad_token_id)).to(dtype=torch.long)
        # seq_length = attention_mask.size(1)
        # eos_indices = seq_length - attention_mask.long().fliplr().argmax(dim=1, keepdim=True).clamp(min=1)
        # sequences.scatter_(dim=1, index=eos_indices, value=eos_token_id)

        # # For Llama3 and Qwen2 models, there are some eos_tokens in the middle of the prompt.
        # first_token_indices = attention_mask.long().argmax(dim=1, keepdim=True)
        # mask = torch.arange(seq_length).unsqueeze(0).expand(sequences.size(0), -1).to(device=sequences.device)
        # attention_mask = (mask >= first_token_indices) & (mask <= eos_indices).to(dtype=torch.long)

        # # in RL, state_i (current token) + action_i (next token) -> state_i+1 (next token)
        # state_seq = sequences[:, input_len - 1 : -1]
        # action_mask = state_seq.ne(eos_token_id) & state_seq.ne(pad_token_id)
        # action_mask[:, 0] = 1

        
        logits = self.reward_model(reencoded_sequences, attention_mask=reencoded_attention_mask)
        logits = logits[..., candidate_tokens]
        scores = logits.softmax(dim=-1)[:,:,0]
        # mask_1 = scores > 0.7
        # scores[mask_1] = 1
        # scores[~mask_1] = -1
        # scores = torch.where(scores > 0.7, torch.tensor(1.0, device=scores.device), torch.tensor(-1.0, device=scores.device))
        scores = scores*2-1

        # rewards = torch.zeros_like(sequences)
        tokens = self.tokenizer.convert_ids_to_tokens(reencoded_sequences[0].tolist())
        sep_mask = torch.zeros_like(sequences, dtype=torch.bool)
        # 批量转换 token IDs 为 tokens
        batch_tokens = [self.tokenizer.convert_ids_to_tokens(ids.tolist()) for ids in sequences]

        # 更新 mask，若 token 中包含 '\n' 或 '.Ċ' 则将对应位置的 mask 设为 True
        for i, tokens in enumerate(batch_tokens):
            tokens_pre = [None, None]
            for j, token in enumerate(tokens):
                # if token is None:
                #     with open('why_none.txt', 'w') as f:
                #         f.write(str(tokens)+'\n')
                #     breakpoint()
                if token is not None and (('\n' in token) or ('Ċ' in token)):
                    # if (last_token is None) or not sep_mask[i,j-1] or (last_token[-1]=='\n' or last_token[-1]=='Ċ') and (token[0]!='\n' or token[0]!='Ċ'):
                    last_token = tokens_pre[1]
                    last_last_token = tokens_pre[0]
                    if (last_token is None) or (not sep_mask[i,j-1]) or ((token[0]!='\n' or token[0]!='Ċ') and not (bool(re.match(r'^[ĠĊĉč]*$', token)))):
                        sep_mask[i, j] = True
                    if last_last_token is not None:
                        # last_last_token_reverse = last_last_token[::-1]
                        # 从右往左最后一个为空白的字符位置
                        # last_last_token_len = len(last_last_token_reverse)
                        last_last_token_first_non_matching_char_idx = self.find_first_matching_char(last_last_token)
                        # 从左往右最后一个为空白的字符位置
                        this_token_reverse = token[::-1]
                        this_token_reverse_first_non_matching_char_idx = self.find_first_matching_char(this_token_reverse)
                        slice_last_last_token = last_last_token[last_last_token_first_non_matching_char_idx+1:]
                        slice_this_token = this_token_reverse[this_token_reverse_first_non_matching_char_idx+1:][::-1]
                        if sep_mask[i, j-2] and bool(re.match(r'^[ĠĉčĊ]*$',slice_last_last_token+last_token+slice_this_token)):
                            sep_mask[i, j] = False
                tokens_pre = [tokens_pre[1], token]
        origin_sep_mask = sep_mask.clone()
        # sep_mask = sep_mask & ~torch.cat([torch.zeros(sep_mask.shape[0], 1, dtype=torch.bool, device=sep_mask.device), sep_mask[:, :-1]], dim=1)
        # sep_mask = sequences == step_separater_id
        rewards = torch.zeros_like(sequences, device=scores.device, dtype=scores.dtype)

        km_mask = (reencoded_sequences == km_token_id1) | (reencoded_sequences == km_token_id2)
        km_mask = km_mask & ~torch.cat([torch.zeros(km_mask.shape[0], 1, dtype=torch.bool, device=km_mask.device), km_mask[:, :-1]], dim=1)
       
        if km_mask.sum().item() != sep_mask.sum().item():
            with open('error.txt', 'w') as f:
                f.write('\n'+'='*100+'\n')
                f.write(f'km_mask: {km_mask.sum().item()}; sep_mask: {sep_mask.sum().item()}\n')
                f.write(f'km_mask shape: {km_mask.shape}; sep_mask shape: {sep_mask.shape}\n')
                f.write(f'origin km_mask: {((reencoded_sequences == km_token_id1) | (reencoded_sequences == km_token_id2)).sum().item()}; ')
                f.write(f'origin sep_mask: {origin_sep_mask.sum().item()}\n')
                for i, tokens in enumerate(batch_tokens):
                    for j in range(len(tokens)):
                        newline_token = ('\n' in tokens[j]) or ('Ċ' in tokens[j])
                        f.write(f'({i},{j}): {tokens[j]}; ac{newline_token}\n')
                    # f.write('\n'.join(tokens))
                    f.write('\n')
                # f.write(str(reencoded_sequences.tolist()))
                f.write('\n')
                critic_batch_tokens = [self.critic_tokenizer.convert_ids_to_tokens(ids.tolist()) for ids in reencoded_sequences]
                for i, tokens in enumerate(critic_batch_tokens):
                    for j in range(len(tokens)):
                        f.write(f'({i},{j}): {tokens[j]}; {reencoded_sequences[i][j]}\n')
                    # f.write('\n'.join(tokens))
                    f.write('\n')
                # f.write
                f.write(str(decoded_sequences))
            breakpoint()
        
        # 验证 km token 对应到 换行token了
        with open('valid_token_map.txt', 'a') as f:
            f.write('\n'+'='*100+'\n')
            critic_batch_tokens = [self.critic_tokenizer.convert_ids_to_tokens(ids.tolist()) for ids in reencoded_sequences]
            # 获取为 True 的元素的索引
            sep_rows, sep_cols = torch.nonzero(sep_mask, as_tuple=True)
            km_rows, km_cols = torch.nonzero(km_mask, as_tuple=True)
            for i in range(len(sep_rows)):
                sep_r, sep_c = sep_rows[i], sep_cols[i]
                km_r, km_c = km_rows[i], km_cols[i]
                # 输出附近的tokens
                f.write(f'{i}: sep({sep_r}, {sep_c}): {batch_tokens[sep_r][sep_c-3:sep_c+3]};\n\tkm({km_r}, {km_c}): {critic_batch_tokens[km_r][km_c-3:km_c+3]};\n\tscore: {scores[km_r][km_c]}\n')
            

        rewards[sep_mask] = scores[km_mask]
        # 找到原来
        # rewards = torch.zeros_like(scores, device=scores.device)
        # mask = reencoded_sequences == km_token_id
        # rewards[mask] = scores[mask]

        # rewards_len = rewards.size(1)
        # if rewards_len < num_actions:
        #     new_rewards = torch.zeros(rewards.size(0), num_actions, device=rewards.device)
        #     new_rewards[:, :rewards_len] = rewards
        #     rewards = new_rewards
        # else:
        #     rewards = rewards[:, -num_actions:]
        if rewards.size(1) > num_actions:
            rewards = rewards[:, -num_actions:]
            sep_mask = sep_mask[:, -num_actions:]
        return rewards, sep_mask