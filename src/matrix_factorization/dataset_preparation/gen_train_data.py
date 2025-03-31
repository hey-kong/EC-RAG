import random
import jsonlines
from transformers import AutoTokenizer

# 输入和输出文件路径
input_file = '/data/dataset/gpt4_dataset.jsonl'
output_file = '/data/dataset/balanced_mf_train_gpt4_dataset.jsonl'

model_path = "BAAI/bge-small-en-v1.5"
# 加载分词器，后续需要过滤掉token数过长的样本
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 读取原始数据集并处理
filtered_cnt = 0
data = []
with jsonlines.open(input_file) as reader:
    for idx, record in enumerate(reader):
        # 提取所需字段
        prompt = record.get('prompt', '')

        tokens = tokenizer(prompt, padding='max_length', return_tensors='pt')
        if tokens['input_ids'].shape[1] > 512:
            filtered_cnt += 1
            continue

        mixtral_score = record.get('mixtral_score', 0)

        # 确定winner
        winner = "model_b" if mixtral_score >= 4 else "model_a"

        # 添加到数据列表
        data.append({
            "prompt": prompt,
            "model_a": "llm",
            "model_b": "slm",
            "winner": winner,
            "idx": idx
        })

        # 输出处理进度
        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1} records...")

print(f"Filtered {filtered_cnt} records due to token length > 512")

# 统计winner的数量
model_a_count = sum(1 for record in data if record['winner'] == 'model_a')
model_b_count = sum(1 for record in data if record['winner'] == 'model_b')

print(f"Original counts - model_a: {model_a_count}, model_b: {model_b_count}")

# 平衡数据集
if model_a_count > model_b_count:
    # 随机剔除多余的model_a样本
    excess = model_a_count - model_b_count
    model_a_records = [record for record in data if record['winner'] == 'model_a']
    random.shuffle(model_a_records)
    to_remove = set([record['idx'] for record in model_a_records[:excess]])
    data = [record for record in data if record['idx'] not in to_remove]
elif model_b_count > model_a_count:
    # 随机剔除多余的model_b样本
    excess = model_b_count - model_a_count
    model_b_records = [record for record in data if record['winner'] == 'model_b']
    random.shuffle(model_b_records)
    to_remove = set([record['idx'] for record in model_b_records[:excess]])
    data = [record for record in data if record['idx'] not in to_remove]

# 重新生成idx
for new_idx, record in enumerate(data):
    record['idx'] = new_idx

# 统计平衡后的winner数量
balanced_model_a_count = sum(1 for record in data if record['winner'] == 'model_a')
balanced_model_b_count = sum(1 for record in data if record['winner'] == 'model_b')

print(f"Balanced counts - model_a: {balanced_model_a_count}, model_b: {balanced_model_b_count}")

# 写入平衡后的数据集
with jsonlines.open(output_file, mode='w') as writer:
    writer.write_all(data)

print(f"Balanced dataset saved to {output_file}")