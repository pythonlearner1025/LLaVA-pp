# This file is modified from https://github.com/haotian-liu/LLaVA/

import argparse
import re
from io import BytesIO
import os, os.path as osp
import json
from tqdm import tqdm

import requests
import torch
from PIL import Image

from llava.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                             DEFAULT_IMAGE_TOKEN, IMAGE_PLACEHOLDER,
                             IMAGE_TOKEN_INDEX)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (KeywordsStoppingCriteria, get_model_name_from_path,
                            process_images, tokenizer_image_token)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(url, img_dir=None):
    if not img_dir:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        img_file = os.path.join(img_dir, url.split('/')[-1])
        image = Image.open(img_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def eval_model(args):
    # Model
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    print(model_name)
    print(args.model_path)
    print(args.model_base)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)

    # Load eval data
    eval_dir = '/home'
    out_dir = '/home'
    img_dir = '/home/reka-vibe-eval/data/imgs'
    eval_data_path = os.path.join(eval_dir, 'reka-vibe-eval/data/vibe-eval.v1.jsonl')
    eval_data = []
    with open(eval_data_path, 'r') as f:
        for line in f:
            eval_data.append(json.loads(line))

    name = 'phi-vision-general.jsonl'
    output_filename = os.path.join(out_dir, name)
    with open(output_filename, 'w') as f:
        progress = tqdm(total=len(eval_data), desc='Evaluating')
        for example in eval_data:
            example_id = example['example_id']
            prompt = example['prompt']
            image_url = example['media_url']

            # Load image
            print('loading image...')
            image = load_image(image_url, img_dir=img_dir)
            images = [image]

            image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            if IMAGE_PLACEHOLDER in prompt:
                if model.config.mm_use_im_start_end:
                    prompt = re.sub(IMAGE_PLACEHOLDER, image_token_se, prompt)
                else:
                    prompt = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, prompt)
            else:
                if DEFAULT_IMAGE_TOKEN not in prompt:
                    print("no <image> tag found in input. Automatically append one at the beginning of text.")
                    # do not repeatively append the prompt.
                    if model.config.mm_use_im_start_end:
                        prompt = (image_token_se + "\n") * len(images) + prompt
                    else:
                        prompt = (DEFAULT_IMAGE_TOKEN + "\n") * len(images) + prompt
            print("input: ", prompt)

            if "llama-2" in model_name.lower():
                conv_mode = "llava_llama_2"
            elif "v1" in model_name.lower():
                conv_mode = "llava_v1"
            elif "mpt" in model_name.lower():
                conv_mode = "mpt"
            else:
                conv_mode = "llava_v0"

            if args.conv_mode is not None and conv_mode != args.conv_mode:
                print(
                    "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                        conv_mode, args.conv_mode, args.conv_mode
                    )
                )
            else:
                args.conv_mode = conv_mode

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            images_tensor = process_images(images, image_processor, model.config).to(model.device, dtype=torch.float16)
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            print("img tensor: ",images_tensor.shape)
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=[
                        images_tensor,
                    ],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                )

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[: -len(stop_str)]
            outputs = outputs.strip()
            # Save the generated output
            pair = {"example_id": example_id, "generation": outputs}
            print(pair)
            f.write(json.dumps(pair) + "\n")
            progress.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="MBZUAI/LLaVA-Phi-3-mini-4k-instruct")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    eval_model(args)
