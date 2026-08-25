import json
import os
import re
from collections import defaultdict


def parse_instruction_and_chosen(instruction, chosen):
    """
    解析instruction和chosen字段，提取system prompt和完整对话
    
    instruction格式示例:
    <|im_start|>system\n请你扮演以下人设：...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n
    
    chosen格式示例:
    回复内容<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n回复内容<|im_end|>...
    
    返回:
    - system_prompt: system prompt内容
    - instruction_dialogues: instruction中的对话（不包括最后一个未完成的assistant）
    - chosen_dialogues: chosen中的对话（包括开头补全的assistant回复）
    """
    
    # 解析system prompt
    system_match = re.search(r'<\|im_start\|>system\n(.*?)<\|im_end\|>', instruction, re.DOTALL)
    system_prompt = system_match.group(1).strip() if system_match else ""
    
    # 移除system部分，只处理user和assistant的对话
    instruction_without_system = re.sub(r'<\|im_start\|>system\n.*?<\|im_end\|>\n?', '', instruction, flags=re.DOTALL)
    
    # 解析instruction中的对话
    # instruction最后通常是 <|im_start|>assistant\n (没有内容，等待chosen来补全)
    instruction_dialogues = []
    
    # 匹配完整的对话轮次 (有<|im_end|>结尾的)
    pattern_complete = r'<\|im_start\|>(user|assistant)\n(.*?)<\|im_end\|>'
    matches = re.findall(pattern_complete, instruction_without_system, re.DOTALL)
    
    for role, content in matches:
        content = content.strip()
        if content:
            instruction_dialogues.append({
                "role": role,
                "content": content,
                "from_instruction": True  # 标记来自instruction
            })
    
    # 解析chosen中的对话
    # chosen开头直接是assistant的回复内容（补全instruction最后的assistant）
    chosen_dialogues = []
    
    # chosen的格式: 回复内容<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n回复<|im_end|>...
    # 首先处理开头的assistant回复（没有<|im_start|>assistant\n前缀）
    first_end_match = re.match(r'^(.*?)<\|im_end\|>', chosen, re.DOTALL)
    if first_end_match:
        first_content = first_end_match.group(1).strip()
        if first_content:
            chosen_dialogues.append({
                "role": "assistant",
                "content": first_content,
                "from_instruction": False  # 标记来自chosen
            })
        
        # 处理剩余部分
        remaining = chosen[first_end_match.end():]
        matches = re.findall(pattern_complete, remaining, re.DOTALL)
        for role, content in matches:
            content = content.strip()
            if content:
                chosen_dialogues.append({
                    "role": role,
                    "content": content,
                    "from_instruction": False  # 标记来自chosen
                })
    
    return system_prompt, instruction_dialogues, chosen_dialogues


def convert_item(item):
    """将单条DPO数据转换为训练格式，只使用chosen部分
    
    - instruction中的对话：user和assistant都是trainable=False
    - chosen中的对话：user是trainable=False，assistant是trainable=True
    """
    
    instruction = item.get("instruction", "")
    chosen = item.get("chosen", "")
    
    # 解析instruction和chosen
    system_prompt, instruction_dialogues, chosen_dialogues = parse_instruction_and_chosen(instruction, chosen)
    
    # 合并所有对话
    all_dialogues = instruction_dialogues + chosen_dialogues
    
    # 构建训练格式
    dialog_list = []
    for idx, msg in enumerate(all_dialogues):
        role = msg["role"]
        speaker = "<|im_start|>" + role
        content = msg["content"] + "<|im_end|>"
        
        if role == "user":
            trainable = False
        else:
            trainable = True
        
        cur_ret = {
            "idx": str(idx),
            "speaker": speaker,
            "content": content,
            "trainable": trainable
        }
        dialog_list.append(cur_ret)
    
    ret = {
        "dynamic_text": [
            {
                "label": "",
                "content": "<|im_start|>system\n" + system_prompt + "<|im_end|>",
                "trainable": False  # system prompt不训练
            }
        ],
        "dialogues": dialog_list
    }
    
    return ret


def main():
    input_path = os.path.join("train_data", "dpo_all_0425.jsonl")
    output_path = os.path.join("train_data", "dpo_multi_turn_0426_longest_train.jsonl")

    print(f"读取文件: {input_path}")

    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    print(f"共 {len(data)} 条数据")

    # 按 system prompt 分组，每组只保留对话轮数最长的
    groups = defaultdict(list)
    for item in data:
        instruction = item.get("instruction", "")
        sys_match = re.search(r'<\|im_start\|>system\n(.*?)<\|im_end\|>', instruction, re.DOTALL)
        sys_prompt = sys_match.group(1).strip() if sys_match else ""
        full_text = instruction + item.get("chosen", "")
        turn_count = len(re.findall(r'<\|im_start\|>assistant', full_text))
        groups[sys_prompt].append((turn_count, item))

    # 每组取最长的
    longest_items = []
    for sys_prompt, items in groups.items():
        items.sort(key=lambda x: x[0], reverse=True)
        longest_items.append(items[0][1])

    print(f"共 {len(groups)} 个独立对话，每个取最长版本，筛选后 {len(longest_items)} 条")

    converted_count = 0
    skipped_count = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(longest_items):
            try:
                ret = convert_item(item)
                if ret["dialogues"]:
                    f.write(json.dumps(ret, ensure_ascii=False) + "\n")
                    converted_count += 1
                else:
                    print(f"警告: 第 {i+1} 条数据没有有效对话，已跳过")
                    skipped_count += 1
            except Exception as e:
                print(f"错误: 第 {i+1} 条数据转换失败: {e}")
                skipped_count += 1

    print(f"转换完成！")
    print(f"  - 成功转换: {converted_count} 条")
    print(f"  - 跳过: {skipped_count} 条")
    print(f"  - 输出文件: {output_path}")


if __name__ == "__main__":
    main()
