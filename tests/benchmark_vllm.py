from llm_utils import OAI_LM
from speedy_utils import multi_thread
lm = OAI_LM(port=8155, temperature=0.95, cache=False)


prompts = ['hello, write a 5000 word essay on the topic of anything']

def f(prompt):
    return lm(prompt, max_tokens=5000)

multi_thread(f, prompts*100, workers=128)