import json
import os
import re
import random
from collections import defaultdict


def parse_instruction_and_response(instruction, response):
    """
    解析instruction和response(chosen或rejected)字段，提取system prompt和完整对话

    返回:
    - system_prompt: system prompt内容
    - instruction_dialogues: instruction中的对话列表
    - response_dialogues: response中的对话列表
    """

    system_match = re.search(r'<\|im_start\|>system\n(.*?)<\|im_end\|>', instruction, re.DOTALL)
    system_prompt = system_match.group(1).strip() if system_match else ""

    instruction_without_system = re.sub(r'<\|im_start\|>system\n.*?<\|im_end\|>\n?', '', instruction, flags=re.DOTALL)

    instruction_dialogues = []
    pattern_complete = r'<\|im_start\|>(user|assistant)\n(.*?)<\|im_end\|>'
    matches = re.findall(pattern_complete, instruction_without_system, re.DOTALL)

    for role, content in matches:
        content = content.strip()
        if content:
            instruction_dialogues.append({
                "role": role,
                "content": content,
            })

    response_dialogues = []
    first_end_match = re.match(r'^(.*?)<\|im_end\|>', response, re.DOTALL)
    if first_end_match:
        first_content = first_end_match.group(1).strip()
        if first_content:
            response_dialogues.append({
                "role": "assistant",
                "content": first_content,
            })

        remaining = response[first_end_match.end():]
        matches = re.findall(pattern_complete, remaining, re.DOTALL)
        for role, content in matches:
            content = content.strip()
            if content:
                response_dialogues.append({
                    "role": role,
                    "content": content,
                })

    return system_prompt, instruction_dialogues, response_dialogues


def build_training_item(system_prompt, dialogues, trainable_indices):
    """
    根据对话列表和trainable位置集合，构建一条训练数据
    """
    dialog_list = []
    for idx, msg in enumerate(dialogues):
        role = msg["role"]
        speaker = "<|im_start|>" + role
        content = msg["content"] + "<|im_end|>"

        trainable = (role == "assistant" and idx in trainable_indices)

        dialog_list.append({
            "idx": str(idx),
            "speaker": speaker,
            "content": content,
            "trainable": trainable
        })

    return {
        "dynamic_text": [
            {
                "label": "",
                "content": "<|im_start|>system\n" + system_prompt + "<|im_end|>",
                "trainable": False
            }
        ],
        "dialogues": dialog_list
    }


def process_group(group_items):
    """
    处理同一个system prompt下的所有数据，返回多条训练数据

    逻辑:
    1. 按total_turns排序（短→长），构成对话链
    2. 对每条数据检查：它的response内容是否和更长数据的instruction中对应位置一致
       - 一致（选了chosen/winner）→ 可以合并进长链，标记该位置trainable
       - 不一致（选了rejected）→ 作为独立短数据输出
    3. 最长的那条数据，无论选了什么，都作为链的骨架输出
    """
    group_items.sort(key=lambda x: x["total_turns"])

    results = []
    system_prompt = group_items[0]["system_prompt"]

    # 最长的item作为骨架
    longest = group_items[-1]
    base_dialogues = longest["instruction_dialogues"] + longest["response_dialogues"]

    # 收集可合并进长链的trainable位置
    merge_trainable_indices = set()

    # 最长item自己的response部分一定算trainable
    for offset, msg in enumerate(longest["response_dialogues"]):
        turn_idx = longest["response_start_idx"] + offset
        if msg["role"] == "assistant":
            merge_trainable_indices.add(turn_idx)

    # 处理较短的items
    for item in group_items[:-1]:
        start_idx = item["response_start_idx"]
        response_dialogues = item["response_dialogues"]

        # 检查该item的response内容是否与base中对应位置一致
        can_merge = True
        for offset, msg in enumerate(response_dialogues):
            turn_idx = start_idx + offset
            if turn_idx < len(base_dialogues):
                if msg["content"] != base_dialogues[turn_idx]["content"]:
                    can_merge = False
                    break
            else:
                can_merge = False
                break

        if can_merge:
            # 选了chosen（跟长链一致），合并进去
            for offset, msg in enumerate(response_dialogues):
                turn_idx = start_idx + offset
                if msg["role"] == "assistant":
                    merge_trainable_indices.add(turn_idx)
        else:
            # 选了rejected（跟长链不一致），独立输出
            standalone_dialogues = item["instruction_dialogues"] + response_dialogues
            standalone_trainable = set()
            for offset, msg in enumerate(response_dialogues):
                turn_idx = start_idx + offset
                if msg["role"] == "assistant":
                    standalone_trainable.add(turn_idx)

            ret = build_training_item(system_prompt, standalone_dialogues, standalone_trainable)
            if ret["dialogues"]:
                results.append(ret)

    # 输出合并后的长链
    ret = build_training_item(system_prompt, base_dialogues, merge_trainable_indices)
    if ret["dialogues"]:
        results.append(ret)

    return results


def main():
    input_path = os.path.join("train_data", "dpo_all_0425.jsonl")
    output_path = os.path.join("train_data", "dpo_random_merge_train.jsonl")

    random.seed(42)

    print(f"读取文件: {input_path}")

    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    print(f"共 {len(data)} 条数据")

    # 解析每条数据，随机选择chosen或rejected
    parsed_items = []
    for item in data:
        instruction = item.get("instruction", "")
        chosen = item.get("chosen", "")
        rejected = item.get("rejected", "")

        # 随机选择chosen或rejected
        if chosen and rejected:
            response = random.choice([chosen, rejected])
        elif chosen:
            response = chosen
        else:
            response = rejected

        if not response:
            continue

        system_prompt, instruction_dialogues, response_dialogues = \
            parse_instruction_and_response(instruction, response)

        if not response_dialogues:
            continue

        parsed_items.append({
            "system_prompt": system_prompt,
            "instruction_dialogues": instruction_dialogues,
            "response_dialogues": response_dialogues,
            "response_start_idx": len(instruction_dialogues),
            "total_turns": len(instruction_dialogues) + len(response_dialogues),
        })

    print(f"有效解析: {len(parsed_items)} 条")

    # 按system prompt分组
    groups = defaultdict(list)
    for item in parsed_items:
        groups[item["system_prompt"]].append(item)

    print(f"共 {len(groups)} 个独立system prompt组")

    converted_count = 0
    merged_count = 0
    standalone_count = 0
    skipped_count = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for sys_prompt, group_items in groups.items():
            try:
                results = process_group(group_items)
                for ret in results:
                    f.write(json.dumps(ret, ensure_ascii=False) + "\n")
                    converted_count += 1
                # 每组至少1条合并的长链，其余是standalone的rejected短数据
                merged_count += 1
                standalone_count += len(results) - 1
            except Exception as e:
                print(f"错误: 合并失败 (system_prompt前20字: {sys_prompt[:20]}): {e}")
                skipped_count += 1

    print(f"转换完成！")
    print(f"  - 总输出: {converted_count} 条")
    print(f"    - 合并长链: {merged_count} 条")
    print(f"    - 独立短数据(rejected): {standalone_count} 条")
    print(f"  - 跳过: {skipped_count} 条")
    print(f"  - 输出文件: {output_path}")


if __name__ == "__main__":
    main()
