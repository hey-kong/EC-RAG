from openai import OpenAI
import time

# Custom module
from customed_statistic import global_statistic

# def query_prompt(chunks, query):
#     chunk_str = "\n\n".join(chunks)

#     prompt = f"""{chunk_str}

# Based on the above information, answer the given question: {query}

# Do not output preamble or explanations.

# Answer:
# """

#     return prompt

def query_prompt(chunk_list, query):
    chunk_str = "\n\n".join(chunk_list)

    prompt_template = f"""{chunk_str}

Based on the above information, only give me the answer and do not output any other words.

Question: {query}

Answer:"""

    return prompt_template


# Load model
model = "gpt-4o-mini"
# api_key = os.getenv("OPENAI_API_KEY")
# if not api_key:
#     api_key = input("Please input OpenAI API Key: ").strip()
api_key = "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"
client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

def generate_answer(chunk_list, query_text):
    prompt = query_prompt(chunk_list, query_text)
    try:
        start_time = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            messages=[
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
    if not response or not response.choices:
        print("错误: 响应无效或choices为空。")
        return "请求无响应。", len(chunk_list)

    first_choice = response.choices[0]
    if not hasattr(first_choice, 'message'):
        print("错误: choice中缺少message字段。")
        return "响应格式异常。", len(chunk_list)

    message_content = first_choice.message.content
    if not message_content:
        print("警告：返回内容为空。")
        message_content = "未获取到有效回答。"

    return message_content, len(chunk_list)