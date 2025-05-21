import os
import time
from openai import OpenAI

from customed_statistic import global_statistic
from utils import calc_cost


def query_prompt(chunk_list, query):
    chunks = "\n\n".join(chunk_list)

    prompt_template = f"""\
{chunks}

Given the above context, answer the question: {query}

Only give me the answer and do not output any other words.
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


def generate_answer(query_text, chunk_list, estimate_cost=False):
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
        print(f"API call exception: {str(e)}")
        return "No valid answer was obtained."

    global_statistic.add_to_list("llm_generate_time", generate_time)

    # Check response
    if not response:
        print("Error: Response is empty.")
        return ""

    if not response.choices:
        print("Error: Response contains no choices.")
        return ""

    first_choice = response.choices[0]
    if not hasattr(first_choice, 'message'):
        print("Error: Missing 'message' field in choice.")
        return ""

    message_content = first_choice.message.content
    if not message_content:
        print("Warning: Response content is empty.")
        message_content = ""

    if estimate_cost:
        cost = calc_cost(prompt, message_content)
        global_statistic.add_to_list("cloud_api_cost", cost)

    return message_content
