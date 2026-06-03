# Copyright 2025 TTRL Team (https://arxiv.org/abs/2504.16084)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List
from collections import Counter
import torch
import numpy as np
from verl.utils.reward_score.ttrl_math import extract_answer, simplify_expression_string, grade

def select_top_k_per_prompt(data, n_votes_per_prompt, n_samples_per_prompt):
    """
    Select the first k rollouts per prompt, used for TTRL downsampling.
    """
    assert len(data) % n_votes_per_prompt == 0, "data length must be divisible by n_votes_per_prompt"
    num_prompts = len(data) // n_votes_per_prompt

    selected_indices = []
    for i in range(num_prompts):
        start = i * n_votes_per_prompt
        selected_indices.extend(range(start, start + n_samples_per_prompt))

    return data[selected_indices]

def select_top_k_per_prompt_cnt(data, n_votes_per_prompt, n_samples_per_prompt, majority_ratio_list, second_ratio_list, bcs=False):
    """
    For each prompt, select K training samples from N rollouts.

    When bcs=True (BCS: Balanced Confidence-Aware Sampling):
      - K+ = min(count(y*), K//2)  top-frequency (positive) samples
      - K- = K - K+                lowest-frequency (negative) samples
      Medium-frequency ambiguous samples are discarded.

    When bcs=False:
      - Falls back to simple top-K selection (highest count).
    """

    # batch[i].meta_info['counter']=counter
    assert len(data) % n_votes_per_prompt == 0
    assert n_samples_per_prompt <= n_votes_per_prompt
    # print(majority_ratio_list)
    num_prompts = len(data) // n_votes_per_prompt
    selected_indices = []

    for i in range(num_prompts):
        start = i * n_votes_per_prompt
        end = start + n_votes_per_prompt
        group = data[start:end]
        # (index, count)
        indexed_group = []
        for local_idx, item in enumerate(group):
            count = item.non_tensor_batch['count']
            indexed_group.append((start + local_idx, count))

        # sort by count (descending)
        indexed_group.sort(key=lambda x: x[1], reverse=True)

        if bcs:
            # BCS: K+ top-frequency positives (capped at K//2) + K- lowest-frequency negatives
            k = n_samples_per_prompt
            k_high = min(int(majority_ratio_list[i] * n_votes_per_prompt), n_samples_per_prompt // 2)
            k_low = k - k_high
            top_part = indexed_group[:k_high]
            bottom_part = indexed_group[-k_low:]
            selected_indices.extend(idx for idx, _ in top_part)
            selected_indices.extend(idx for idx, _ in bottom_part)
        else:
            # Default: top-K by frequency
            top_part = indexed_group[:n_samples_per_prompt]
            selected_indices.extend(idx for idx, _ in top_part)

    return data[selected_indices]


# === Ground Truth Manipulation ===


def apply_original_gt(batch):
    """
    Apply the original ground truth to the batch.
    """
    for i in range(len(batch)):
        data_item = batch[i]
        original_gt = data_item.non_tensor_batch["reward_model"]["original_gt"]
        data_item.non_tensor_batch["reward_model"]["ground_truth"] = original_gt

    return batch


def apply_ttrl_gt(batch, gen_batch_output, n, tokenizer):
    """
    Apply the majority vote ground truth to the batch.
    """
    assert len(gen_batch_output) % n == 0, "gen_batch_output length must be divisible by n"
    num_prompts = len(gen_batch_output) // n
    assert len(batch) == num_prompts, "batch length must be equal to the number of prompts"

    model_outputs = []  
    for i in range(num_prompts):
        start = i * n
        for j in range(n):
            data_item = gen_batch_output[start + j]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            model_outputs.append(response_str)

    majority_gt_list, majority_ratio_list, second_ratio_list = _batch_majority_vote(model_outputs, n)
    
    assert len(batch) == len(majority_gt_list), "batch length must be equal to the number of model outputs"
    
    for i in range(num_prompts):
        data_item = batch[i]
        original_gt = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        data_item.non_tensor_batch["reward_model"]["ground_truth"] = majority_gt_list[i]
        data_item.non_tensor_batch["reward_model"]["majority_gt"] = majority_gt_list[i]
        data_item.non_tensor_batch["reward_model"]["original_gt"] = original_gt
    
    batch.non_tensor_batch["majority_ratio_list"] = np.array(majority_ratio_list, dtype=float)
    batch.non_tensor_batch["second_ratio_list"] = np.array(second_ratio_list, dtype=float)
    return batch

def apply_ttrl_cnt(batch, gen_batch_output, n, tokenizer):
    """
    For each generated sample, compute how many times its (extracted) answer
    appears in its prompt group, and store it in data_item.batch["count"].
    
    Returns:
        batch, gen_batch_output
    """
    assert len(gen_batch_output) % n == 0, "gen_batch_output length must be divisible by n"
    num_prompts = len(gen_batch_output) // n
    assert len(batch) == num_prompts, "batch length must match num_prompts"

    # Step 1: decode answers for every sample (same logic as apply_ttrl_gt)
    decoded_answers = []
    for i in range(num_prompts):
        start = i * n
        for j in range(n):
            data_item = gen_batch_output[start + j]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            decoded_answers.append(response_str)

    # Step 2: extract & normalize answers
    extracted_answers = []
    for ans in decoded_answers:
        a = extract_answer(ans)
        if a is None:
            extracted_answers.append(None)
        else:
            extracted_answers.append(simplify_expression_string(a))

    # Step 3: for each prompt group, compute the counts
    counts = []
    for i in range(num_prompts):
        # model outputs for this prompt
        group_answers = extracted_answers[i * n:(i + 1) * n]

        # count frequency
        counter = Counter(group_answers)

        # Step 4: write count back to gen_batch_output
        for j in range(n):
            ans = group_answers[j]
            # if answer is None, count = 0
            count = counter[ans] if ans in counter and ans is not None else 0
            counts.append(count)
    gen_batch_output.non_tensor_batch['count'] = np.array(counts, dtype=object)


    return batch, gen_batch_output


def _batch_majority_vote(model_outputs: List[str], n: int) -> tuple[List[str], List[float], List[float]]:
    """
    Used to generate the ground truth for TTRL.
    Input:
        model_outputs: list of str
        n: int
    Output:
        majority_gt_list: list of str
        majority_ratio_list: list of float
    """
    majority_gt_list = []
    majority_ratio_list = []
    second_ratio_list = []
    assert len(model_outputs) % n == 0
    n_prompts = len(model_outputs) // n
    for i in range(n_prompts):
        prompt_outputs = model_outputs[i * n:(i + 1) * n]
        prompt_majority_gt, prompt_majority_ratio, second_ratio = _majority_vote(prompt_outputs)
        majority_gt_list.append(prompt_majority_gt)
        majority_ratio_list.append(prompt_majority_ratio)
        second_ratio_list.append(second_ratio)
        
    return majority_gt_list, majority_ratio_list, second_ratio_list


# def _majority_vote(model_outputs: List[str]) -> tuple[str, float]:
#     assert len(model_outputs) > 0
#     model_answers = [extract_answer(generated_text) for generated_text in model_outputs]
#     model_answers = [answer for answer in model_answers if answer is not None]
#     model_answers = [simplify_expression_string(answer) for answer in model_answers]
#     if len(model_answers) == 0:
#         return "None", 0.0
    
#     counter = Counter(model_answers)
    
#     majority_answer, majority_count = counter.most_common(1)[0]
#     majority_ratio = majority_count / len(model_outputs)
    
#     return majority_answer, majority_ratio
def _majority_vote(model_outputs: List[str]) -> tuple[str, float, float]:
    assert len(model_outputs) > 0

    model_answers = [extract_answer(generated_text) for generated_text in model_outputs]
    model_answers = [answer for answer in model_answers if answer is not None]
    model_answers = [simplify_expression_string(answer) for answer in model_answers]

    if len(model_answers) == 0:
        return "None", 0.0, 0

    counter = Counter(model_answers)
    most_common = counter.most_common()

    # 第一多
    majority_answer, majority_count = most_common[0]
    majority_ratio = majority_count / len(model_outputs)

    # 第二多（可能不存在）
    if len(most_common) > 1:
        _, second_count = most_common[1] 
        second_ratio = second_count/ len(model_outputs)
    else:
        second_ratio = 0.0

    return majority_answer, majority_ratio, second_ratio


# === Metrics Computation ===


def compute_ttrl_metrics(batch, n):
    """
    Compute the TTRL metrics.
    """
    assert len(batch) % n == 0, "batch length must be divisible by n"
    num_prompts = len(batch) // n

    # Sort the batch by the ID
    idx = sorted(range(len(batch)), key=lambda x: batch[x].non_tensor_batch["extra_info"]["index"])

    majority_reward = []
    gt_reward = []
    majority_label = []
    gt_label = []

    for i in range(len(batch)):
        data_item = batch[idx[i]]
        majority_reward.append(data_item.batch["token_level_scores"].sum().item())
        gt_reward.append(data_item.batch["token_level_scores_original"].sum().item())
        majority_label.append(data_item.non_tensor_batch["reward_model"]["majority_gt"])
        gt_label.append(data_item.non_tensor_batch["reward_model"]["original_gt"]) 

    ttrl_metrics = _batch_compute_ttrl_metrics(majority_reward, gt_reward, majority_label, gt_label, n=n)
    majority_ratio_list = batch.non_tensor_batch["majority_ratio_list"]
    majority_ratio = sum(majority_ratio_list) / len(majority_ratio_list)
    ttrl_metrics["majority_ratio"] = majority_ratio

    return ttrl_metrics


def _batch_compute_ttrl_metrics(
    majority_reward: List[float],
    gt_reward: List[float],
    majority_label: List[str],
    gt_label: List[str],
    n: int,
):
    """
    Compute the TTRL metrics for batch inputs.
    """
    assert len(majority_reward) == len(gt_reward) == len(majority_label) == len(gt_label)
    assert len(majority_reward) % n == 0
    n_prompts = len(majority_reward) // n
    ttrl_metrics = []
    for i in range(n_prompts):
        prompt_majority_reward = majority_reward[i * n:(i + 1) * n]
        prompt_gt_reward = gt_reward[i * n:(i + 1) * n]
        prompt_majority_label = majority_label[i * n:(i + 1) * n]
        prompt_gt_label = gt_label[i * n:(i + 1) * n]

        assert Counter(prompt_majority_label).most_common(1)[0][1] == n
        assert Counter(prompt_gt_label).most_common(1)[0][1] == n

        prompt_majority_label = prompt_majority_label[0]
        prompt_gt_label = prompt_gt_label[0]

        ttrl_metric = _prompt_compute_ttrl_metrics(prompt_majority_reward, prompt_gt_reward, prompt_majority_label, prompt_gt_label)
        ttrl_metrics.append(ttrl_metric)

    # Compute the average metrics
    ttrl_metrics = {k: sum(d[k] for d in ttrl_metrics) / len(ttrl_metrics) for k in ttrl_metrics[0]}

    return ttrl_metrics

def _prompt_compute_ttrl_metrics(
    majority_reward: List[float],
    gt_reward: List[float],
    majority_label: str,
    gt_label: str,
    ):    
    assert len(majority_reward) == len(gt_reward)

    hit_rate = 1.0 if grade(majority_label, gt_label) else 0.0    
    rewards_hit_rate = 0
    for estimate_reward, true_reward in zip(majority_reward, gt_reward):
        if estimate_reward == true_reward:
            rewards_hit_rate += 1
    rewards_hit_rate = rewards_hit_rate / len(majority_reward)
    
    ttrl_metric = {
        "label_accuracy": hit_rate,
        "reward_accuracy": rewards_hit_rate,
        "majority_voting_reward": sum(majority_reward) / len(majority_reward),
        "ground_truth_reward": sum(gt_reward) / len(gt_reward),
        f"pass@{len(majority_reward)}": 1.0 if sum(gt_reward) >= 1 else 0.0,
    }
    return ttrl_metric