import os
import time
from openai import OpenAI

# custom modules
from customed_statistic import global_statistic
from utils import calc_cost


def query_prompt(chunk_list, query):
    chunks = "\n\n".join(chunk_list)

    prompt_template = f"""\
{chunks}

Given the above information and not prior knowledge, answer the question.

Question: {query}

Respond with a concise answer only, do not output any other words.
"""

    return prompt_template


# Load model

# gpt-4o-mini
# model = "gpt-4o-mini"
# api_key = os.getenv("OPENAI_API_KEY")
# client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

# deepseek-v3
model = "deepseek-chat"
client = OpenAI(api_key=os.environ.get("LLM_API_KEY"), base_url="https://api.deepseek.com")


def generate_answer(chunk_list, query_text, estimate_cost=False):
    prompt = query_prompt(chunk_list, query_text)
    try:
        start_time = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {'role': 'user', 'content': prompt}
            ],
            stream=False
        )
        generate_time = time.perf_counter() - start_time
    except Exception as e:
        print(f"API调用异常: {str(e)}")
        return "未获取到有效回答。", len(chunk_list)

    # print(f"Generation time: {generate_time:.6f} seconds")
    global_statistic.add_to_list("generate_time", generate_time)

    # check response
    message_content = ""
    if not response:
        print("错误: 响应为空。")
        return "", len(chunk_list)

    if not response.choices:
        print("错误: choices为空。")
        return "", len(chunk_list)

    first_choice = response.choices[0]
    if not hasattr(first_choice, 'message'):
        print("错误: choice中缺少message字段。")
        return "", len(chunk_list)

    message_content = first_choice.message.content
    if not message_content:
        print("警告：返回内容为空。")
        message_content = ""

    if estimate_cost:
        cost = calc_cost(prompt, message_content)
        global_statistic.add_to_list("cloud_api_cost", cost)

    return message_content, len(chunk_list)
