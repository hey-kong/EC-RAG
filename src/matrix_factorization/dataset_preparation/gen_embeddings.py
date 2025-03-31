import json
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch

# 配置路径
model_path = "BAAI/bge-small-en-v1.5"
jsonl_path = "/data/dataset/balanced_mf_train_gpt4_dataset.jsonl"
output_path = "/data/dataset/balanced_mf_train_embedding.npy"

# 设置计算设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path).to(device)
model.eval()

embeddings = []

with open(jsonl_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        # 解析JSON行数据
        data = json.loads(line.strip())
        prompt = data['prompt']

        # 添加指令前缀（可选）
        # prompt = "Represent this sentence for searching relevant passages: " + prompt

        encoded_input = tokenizer(
            prompt,
            max_length=512,
            padding='max_length',
            return_tensors='pt'
        ).to(device)

        # 检查输入长度
        input_length = encoded_input['input_ids'].shape[1]
        if input_length > 512:
            print(f"警告：第 {i} 条数据截断后仍有 {input_length} tokens")

        # 生成嵌入向量
        with torch.no_grad():
            model_output = model(**encoded_input)

        # 提取并归一化CLS向量
        cls_embedding = model_output.last_hidden_state[:, 0, :]
        cls_embedding = torch.nn.functional.normalize(cls_embedding, p=2, dim=1)

        # 转换格式并存储
        embeddings.append(cls_embedding.cpu().numpy()[0])

# 保存为numpy文件
embeddings_np = np.array(embeddings)
np.save(output_path, embeddings_np)

print(f"嵌入文件已保存至 {output_path}，共处理 {len(embeddings_np)} 条数据。")