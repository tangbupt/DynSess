import json
import os


def convert_item(item):
    """将单条数据转换为训练格式"""
    conversations = item["conversations"]
    
    persona = ""
    chat_history = []
    
    for msg in conversations:
        if msg["from"] == "system":
            persona = msg["value"]
        else:
            chat_history.append({
                "role": msg["from"],
                "content": msg["value"]
            })
    
    dialogs = []
    for idx, sent in enumerate(chat_history):
        role = sent["role"]
        speaker = "<|im_start|>" + role
        content = sent["content"] + "<|im_end|>"
        
        # user说的不训练，assistant说的训练
        trainable = (role == "assistant")
        
        cur_ret = {
            "idx": str(idx),
            "speaker": speaker,
            "content": content,
            "trainable": trainable
        }
        dialogs.append(cur_ret)
    
    ret = {
        "dynamic_text": [
            {
                "label": "",
                "content": "<|im_start|>" + persona + "<|im_end|>",
                "trainable": False  # persona不训练
            }
        ],
        "dialogues": dialogs
    }
    
    return ret


def main():
    input_path = os.path.join("train_data", "sft_0424.jsonl")
    output_path = os.path.join("train_data", "sft_2000persona_0426_context_train.jsonl")
    
    print(f"读取文件: {input_path}")
    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    print(f"共 {len(data)} 条数据，开始转换...")

    with open(output_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(data):
            ret = convert_item(item)
            f.write(json.dumps(ret, ensure_ascii=False) + "\n")
    
    print(f"转换完成，输出文件: {output_path}")


if __name__ == "__main__":
    main()
