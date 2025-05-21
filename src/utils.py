import transformers

chat_tokenizer_dir = "../deepseek_v3_tokenizer"

ds_tokenizer = transformers.AutoTokenizer.from_pretrained(
    chat_tokenizer_dir, trust_remote_code=True
)


# deepseek v3 cost
def calc_cost(
        input: str,
        output: str,
        input_price: float = 0.27,
        output_price: float = 1.1, ) -> float:
    input_cost = input_price * len(ds_tokenizer.encode(input)) / 1000000
    output_cost = output_price * len(ds_tokenizer.encode(output)) / 1000000
    return input_cost + output_cost
