
# Setup
python3.8-intel64 -m venv ~/code/virtualenvs/llm_inference_estimator
source ~/code/virtualenvs/llm_inference_estimator/bin/activate
pip install pip-tools==7.3.0
pip-compile requirements.in
pip-sync requirements.txt

# Links for Carlos
Fundamentals, maybe too theoretical: 
https://kipp.ly/transformer-inference-arithmetic/#latency-calculations

Inference:
https://www.jinghong-chen.net/estimate-vram-usage-in-llm-inference/
https://linden-li.github.io/posts/inference-slides?ref=jinghong-chen.net

For training:
https://blog.eleuther.ai/transformer-math/

Some tools out there:
https://kipp.ly/transformer-inference-arithmetic/#latency-calculations
https://github.com/adarshxs/TokenTally/tree/main

How TGI does its calculations:
https://github.com/huggingface/text-generation-inference/blob/59b3ffea1442e40b37ba92e84687ea9784212556/server/text_generation_server/models/flash_causal_lm.py#L790

Alternative to TGI, this repo has better documented code and are the creators of the algorithms. TGI just "copies" them over:
https://github.com/vllm-project/vllm

Issues I have open with TGI:
https://github.com/huggingface/text-generation-inference/issues/1875
https://github.com/huggingface/text-generation-inference/issues/1863

IMPORTANT:
Communication times between gpus increase latency quite a bit. In mistral7b for bs=1 in=100 out=4 it goes from 15.90ms
with 1gpu to 220.90ms with 2 gpus