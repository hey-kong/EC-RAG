import os
import time
from openai import OpenAI

# custom modules
from customed_statistic import global_statistic
from utils import calc_cost


def query_prompt(chunk_list, query):
    chunks = "\n\n".join(chunk_list)

    prompt_template = f"""{chunks}

Based on the above information, give me the answer and do not output any other words.

Question: {query}

Answer:"""

    return prompt_template


# Load model

# gpt-4o-mini
# model = "gpt-4o-mini"
# api_key = os.getenv("OPENAI_API_KEY")
# client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

# deepseek-v3 火山引擎
model = "ep-20250208150353-d4wsv"
client = OpenAI(
    # 此为默认路径，您可根据业务所在地域进行配置
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    # 从环境变量中获取您的 API Key
    api_key=os.environ.get("ARK_API_KEY"),
)


def generate_answer(chunk_list, query_text, estimate_cost=False):
    prompt = query_prompt(chunk_list, query_text)
    try:
        start_time = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an AI assistant."},
                {'role': 'user',
                 'content': prompt}
            ],
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
