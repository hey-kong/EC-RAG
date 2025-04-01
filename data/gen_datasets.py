import json
import os
from datasets import load_dataset

datasets = ["hotpotqa", "qasper", "narrativeqa", "2wikimqa"]

for dataset in datasets:
    data = load_dataset('THUDM/LongBench', dataset, split='test')

    base_dir = dataset
    os.makedirs(os.path.join(base_dir, 'documents'), exist_ok=True)

    question_dir = os.path.join(base_dir, 'questions')
    answer_dir = os.path.join(base_dir, 'answers')

    os.makedirs(question_dir, exist_ok=True)
    os.makedirs(answer_dir, exist_ok=True)

    with open(os.path.join(question_dir, 'questions.jsonl'), 'w', encoding='utf-8') as q_file, \
         open(os.path.join(answer_dir, 'answers.jsonl'), 'w', encoding='utf-8') as a_file:

        q_buffer = []
        a_buffer = []

        doc_i = 0
        for idx, sample in enumerate(data):
            context = sample.get('context')
            if context:
                doc_i += 1
                with open(os.path.join(base_dir, 'documents', f'doc_{doc_i}.txt'), 'w', encoding='utf-8') as doc_file:
                    doc_file.write(context)

            query = sample.get('input')
            if query:
                q_buffer.append(json.dumps({"id": idx, "query": query}, ensure_ascii=False))

            answers = sample.get('answers', [])
            if answers:
                a_buffer.append(json.dumps({"id": idx, "answers": answers}, ensure_ascii=False))

        if q_buffer:
            q_file.write("\n".join(q_buffer) + "\n")
        if a_buffer:
            a_file.write("\n".join(a_buffer) + "\n")
