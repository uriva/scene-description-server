import logging
from io import BytesIO

import gamla
import requests
import torch
from PIL import Image
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
)


def load_model():
    from llava.model import LlavaLlamaForCausalLM

    model_path = "4bit/llava-v1.5-13b-3GB"
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        device_map="auto",
        load_in_4bit=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
    )
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device="cuda")
    return (
        model,
        AutoTokenizer.from_pretrained(model_path, use_fast=False),
        vision_tower.image_processor,
    )


def _caption_image(model, tokenizer, image_processor, image_file, prompt):
    from llava.constants import (
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IMAGE_TOKEN,
        IMAGE_TOKEN_INDEX,
    )
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import (
        KeywordsStoppingCriteria,
        tokenizer_image_token,
    )
    from llava.utils import disable_torch_init

    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    disable_torch_init()
    conv_mode = "llava_v0"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles
    image_tensor = (
        image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
        .half()
        .cuda()
    )
    inp = f"{roles[0]}: {prompt}"
    inp = (
        DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + inp
    )
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    raw_prompt = conv.get_prompt()
    input_ids = (
        tokenizer_image_token(
            raw_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        .unsqueeze(0)
        .cuda()
    )
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )
    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1] :]).strip()
    conv.messages[-1][-1] = outputs
    output = outputs.rsplit("</s>", 1)[0]
    if not output:
        raise ValueError("Model returned empty output")
    return output


@gamla.timeit
def work_on_file(model, image_path: str, prompt: str):
    logging.info(image_path)
    try:
        return _caption_image(*model, image_path, prompt)
    except Exception as e:
        logging.error(e)
        return None
