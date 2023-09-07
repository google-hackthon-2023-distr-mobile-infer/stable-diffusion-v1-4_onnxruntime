import os
from diffusers import OnnxStableDiffusionPipeline, OnnxRuntimeModel
from diffusers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler, DPMSolverMultistepScheduler
from transformers import CLIPTextModel, CLIPTokenizer
 #https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img
from typing import Callable, List, Optional, Union
from tqdm.auto import tqdm
import numpy as np
import torch
from diffusers.utils.pil_utils import numpy_to_pil
import time
from diffusers.utils import deprecate, logging
logger = logging.get_logger(__name__)

from client import get_task_input,submit_task_result,regist_node

from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=1)


    
############################################################################################################
model_dir = "/Users/zhangyixin/Desktop/hackathon-google/stable-diffusion-v1-4_onnxruntime/"
prompt = "a photo of an astronaut riding a horse on mars"
num_inference_steps = 20
height=64
width=64
# 手机和嵌入shi client上跑tokenizer，text_encoder，vae_decoder
scheduler = PNDMScheduler.from_pretrained(os.path.join(model_dir, "scheduler/scheduler_config.json"))

tokenizer = CLIPTokenizer.from_pretrained(model_dir, subfolder="tokenizer")

text_encoder = OnnxRuntimeModel(model=OnnxRuntimeModel.load_model(os.path.join(model_dir, "text_encoder/model.onnx")))

vae_decoder = OnnxRuntimeModel(model=OnnxRuntimeModel.load_model(os.path.join(model_dir, "vae_decoder/model.onnx")))

#放到了3060gpu服务器上？
# 这里循环20次 ，encoder 会参与吗？
# unet = OnnxRuntimeModel(model=OnnxRuntimeModel.load_model(os.path.join(model_dir, "unet/model.onnx")))
unet = None
#  image = pipe(prompt, guidance_scale=7.5, num_inference_steps=15, generator=generator).images[0]
#第一个手机端的pipe
pipe = OnnxStableDiffusionPipeline(
    vae_encoder=None,
    vae_decoder=vae_decoder,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=scheduler,
    safety_checker=None,
    feature_extractor=None,
    requires_safety_checker=False,
)

#把以下的代码写到main 函数里 
def main():
    
    ########################################################################################################################
    local_algo_type = 'tokenize'
    regist_node(local_algo_type)
    print('完成注册，node_id: ')

    while True:
        input = get_task_input(local_algo_type)
        if not input:
            time.sleep(1)
        if input:

            # 这里是算法的核心逻辑
            prompt = input[0]
            prompt = prompt.split('\n')[0]
            #需要接收 手机上 生成的prompt，tokenizer，text_encoder
            def _encode_prompt(
                prompt: Union[str, List[str]],
                num_images_per_prompt: Optional[int],
                do_classifier_free_guidance: bool,
                negative_prompt: Optional[str],
                prompt_embeds: Optional[np.ndarray] = None,
                negative_prompt_embeds: Optional[np.ndarray] = None,
            ):
                if prompt is not None and isinstance(prompt, str):
                    batch_size = 1
                elif prompt is not None and isinstance(prompt, list):
                    batch_size = len(prompt)
                else:
                    batch_size = prompt_embeds.shape[0]

                if prompt_embeds is None:
                    # get prompt text embeddings
                    text_inputs = pipe.tokenizer(
                        prompt,
                        padding="max_length",
                        max_length=pipe.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="np",
                    )
                    text_input_ids = text_inputs.input_ids
                    untruncated_ids = pipe.tokenizer(prompt, padding="max_length", return_tensors="np").input_ids

                    if not np.array_equal(text_input_ids, untruncated_ids):
                        removed_text = pipe.tokenizer.batch_decode(
                            untruncated_ids[:, pipe.tokenizer.model_max_length - 1 : -1]
                        )
                        logger.warning(
                            "The following part of your input was truncated because CLIP can only handle sequences up to"
                            f" {pipe.tokenizer.model_max_length} tokens: {removed_text}"
                        )

                    prompt_embeds = pipe.text_encoder(input_ids=text_input_ids.astype(np.int32))[0]

                prompt_embeds = np.repeat(prompt_embeds, num_images_per_prompt, axis=0)

                # get unconditional embeddings for classifier free guidance
                if do_classifier_free_guidance and negative_prompt_embeds is None:
                    uncond_tokens: List[str]
                    if negative_prompt is None:
                        uncond_tokens = [""] * batch_size
                    elif type(prompt) is not type(negative_prompt):
                        raise TypeError(
                            f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                            f" {type(prompt)}."
                        )
                    elif isinstance(negative_prompt, str):
                        uncond_tokens = [negative_prompt] * batch_size
                    elif batch_size != len(negative_prompt):
                        raise ValueError(
                            f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                            f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                            " the batch size of `prompt`."
                        )
                    else:
                        uncond_tokens = negative_prompt

                    max_length = prompt_embeds.shape[1]
                    uncond_input = pipe.tokenizer(
                        uncond_tokens,
                        padding="max_length",
                        max_length=max_length,
                        truncation=True,
                        return_tensors="np",
                    )
                    negative_prompt_embeds = pipe.text_encoder(input_ids=uncond_input.input_ids.astype(np.int32))[0]

                if do_classifier_free_guidance:
                    negative_prompt_embeds = np.repeat(negative_prompt_embeds, num_images_per_prompt, axis=0)

                    # For classifier free guidance, we need to do two forward passes.
                    # Here we concatenate the unconditional and text embeddings into a single batch
                    # to avoid doing two forward passes
                    prompt_embeds = np.concatenate([negative_prompt_embeds, prompt_embeds])

                return prompt_embeds
            ############################################################
            #手机上client prompt_embeds需要传给server
            ############################################################
            negative_prompt = None
            prompt_embeds = None
            negative_prompt_embeds = None
            num_images_per_prompt = 1
            do_classifier_free_guidance = True
            prompt_embeds = _encode_prompt(
                prompt,
                num_images_per_prompt,
                do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
            )    
            #prompt_embeds 变成数组,转成json 发给服务端
            prompt_embeds = [prompt_embeds.tolist()]
            submit_task_result(prompt_embeds)

            ############################################################
            # 一直等待client 收到 latents 
            ############################################################
            # latents = None
            # # image = self.vae_decoder(latent_sample=latents)[0]
            # # it seems likes there is a strange result for using half-precision vae decoder if batchsize>1
            # image = np.concatenate(
            #     [pipe.vae_decoder(latent_sample=latents[i : i + 1])[0] for i in range(latents.shape[0])]
            # )

            # image = np.clip(image / 2 + 0.5, 0, 1)
            # image = image.transpose((0, 2, 3, 1))
            # image = numpy_to_pil(image)
            # image[0].save(f"andriod-generated_image_11111.png")

def decoder():
    local_algo_type = 'decode'
    regist_node(local_algo_type)
    while True:
        input = get_task_input(local_algo_type)
        if not input:
            time.sleep(1)
            continue
        ############################################################
        # 一直等待client 收到 latents 
        ############################################################
        latents = input[0]
        latents = np.float32(latents)
        # image = self.vae_decoder(latent_sample=latents)[0]
        # it seems likes there is a strange result for using half-precision vae decoder if batchsize>1
        image = np.concatenate(
            [pipe.vae_decoder(latent_sample=latents[i : i + 1])[0] for i in range(latents.shape[0])]
        )

        image = np.clip(image / 2 + 0.5, 0, 1)
        image = image.transpose((0, 2, 3, 1))
        image = numpy_to_pil(image)
        image[0].save(f"android-generated_image_11111.png")
        # 将PIL图像转换为base64编码 
        import base64 
        
        with open("android-generated_image_11111.png", "rb") as image_file: 
            encoded_string = base64.b64encode(image_file.read())
            submit_task_result(encoded_string.decode('utf-8'))


if __name__ == '__main__':
    import sys
    if sys.argv[-1] == '0':
        main()
    else:
        decoder()
    # main()
    # executor.submit(main)
    # decoder()
