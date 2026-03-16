from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn
import json
import datetime
import torch

DEVICE = "cuda" 
DEVICE_ID = "7" 
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect() 

app = FastAPI()

@app.post("/")
async def create_item(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')
    history = json_post_list.get('history')
    max_length = json_post_list.get('max_length', 2048)
    top_p = json_post_list.get('top_p', 0.7)
    temperature = json_post_list.get('temperature', 0.95)

    messages = []
    if history:
        for h in history:
            messages.append({"role": "user", "content": h[0]})
            messages.append({"role": "assistant", "content": h[1]})
    messages.append({"role": "user", "content": prompt})

    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_length,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
        )

    response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)

    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")

    answer = {
        "response": response,
        "history": history + [[prompt, response]],
        "status": 200,
        "time": time
    }

    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(log)
    torch_gc()
    return answer


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("/data1/guanren/ZhipuAI/glm-4-9b-chat", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "/data1/guanren/ZhipuAI/glm-4-9b-chat",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()

    uvicorn.run(app, host='0.0.0.0', port=8001, workers=1)