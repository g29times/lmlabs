# 真实问题推理
import os
import torch
from modelscope import AutoTokenizer
from transformers import AutoModelForCausalLM

def infer_identity(model, tokenizer):
    # 定义询问模型身份的消息
    messages = [
        {
            "instruction": "你是谁？",
            "input": "",
            "output": ""
        }
    ]
    
    # 准备输入
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    # 生成模型的回答
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=50
    )
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return response

def main():
    model_id = "qwen/Qwen2-1.5B-Instruct"
    model_dir = "./qwen/Qwen2-1___5B-Instruct"

    # 下载并加载模型
    model_dir = snapshot_download(model_id, cache_dir="./", revision="master")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.bfloat16)
    model.enable_input_require_grads()

    # 推理模型身份
    identity = infer_identity(model, tokenizer)
    print("模型的自我识别回答:", identity)

if __name__ == "__main__":
    main()