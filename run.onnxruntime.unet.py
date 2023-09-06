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

from diffusers.utils import deprecate, logging
logger = logging.get_logger(__name__)

model_dir = "/Users/zhangyixin/Desktop/hackathon-google/stable-diffusion-v1-4_onnxruntime/"
 
prompt = "a photo of an astronaut riding a horse on mars"
 

def check_inputs(
    prompt: Union[str, List[str]],
    height: Optional[int],
    width: Optional[int],
    callback_steps: int,
    negative_prompt: Optional[str] = None,
    prompt_embeds: Optional[np.ndarray] = None,
    negative_prompt_embeds: Optional[np.ndarray] = None,
):
    if height % 8 != 0 or width % 8 != 0:
        raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

    if (callback_steps is None) or (
        callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
    ):
        raise ValueError(
            f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
            f" {type(callback_steps)}."
        )

    if prompt is not None and prompt_embeds is not None:
        raise ValueError(
            f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
            " only forward one of the two."
        )
    elif prompt is None and prompt_embeds is None:
        raise ValueError(
            "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
        )
    elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
        raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

    if negative_prompt is not None and negative_prompt_embeds is not None:
        raise ValueError(
            f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
            f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
        )

    if prompt_embeds is not None and negative_prompt_embeds is not None:
        if prompt_embeds.shape != negative_prompt_embeds.shape:
            raise ValueError(
                "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                f" {negative_prompt_embeds.shape}."
            )


#把以下的代码写到main 函数里 
def main():
    num_inference_steps = 20
    height=512
    width=512
    ########################################################################################################################
    # 手机和嵌入上上跑tokenizer，text_encoder，vae_decoder
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
    # # 只为测试使用
    # image = pipe(prompt, num_inference_steps=num_inference_steps, height=height, width=width).images[0] 
    # # # image = pipe(prompt, num_inference_steps=num_inference_steps, height=1024, width=1024).images[0]
    # image.save(f"generated_image_--11111111111111.png")

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

    ########################################################################################################################
    # 服务器上跑unet，是3060笔记本，
    # 构建第二个pipe2 ,使用原来的配置，不用第一个pipe里的unet ；
    #用第二个pipe2 的unet 进行网络推理
    scheduler2 = PNDMScheduler.from_pretrained(os.path.join(model_dir, "scheduler/scheduler_config.json"))
    tokenizer2 = CLIPTokenizer.from_pretrained(model_dir, subfolder="tokenizer")
    text_encoder2 = OnnxRuntimeModel(model=OnnxRuntimeModel.load_model(os.path.join(model_dir, "text_encoder/model.onnx")))
    vae_decoder2 = OnnxRuntimeModel(model=OnnxRuntimeModel.load_model(os.path.join(model_dir, "vae_decoder/model.onnx")))
    unet2 = OnnxRuntimeModel(model=OnnxRuntimeModel.load_model(os.path.join(model_dir, "unet/model.onnx")))
    pipe2 = OnnxStableDiffusionPipeline(
        vae_encoder=None,
        vae_decoder=vae_decoder2,
        text_encoder=text_encoder2,
        tokenizer=tokenizer2,
        unet=unet2,
        scheduler=scheduler2,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )

    ########################################################################################################################
    #server 上跑unet
    ########################################################################################################################
    num_inference_steps=num_inference_steps

    # define call parameters
    batch_size = 1

    generator = np.random

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = True
    num_images_per_prompt = 1


    # get the initial random noise unless the user supplied it
    latents_dtype = prompt_embeds.dtype
    latents_shape = (batch_size * num_images_per_prompt, 4, height // 8, width // 8)

    latents = generator.randn(*latents_shape).astype(latents_dtype)

    # set timesteps
    pipe2.scheduler.set_timesteps(num_inference_steps)

    latents = latents * np.float64(pipe2.scheduler.init_noise_sigma)

    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
    # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
    # and should be between [0, 1]
    accepts_eta = False
    extra_step_kwargs = {}

    timestep_dtype = next(
        (input.type for input in pipe2.unet.model.get_inputs() if input.name == "timestep"), "tensor(float)"
    )

    ORT_TO_NP_TYPE = {
    "tensor(bool)": np.bool_,
    "tensor(int8)": np.int8,
    "tensor(uint8)": np.uint8,
    "tensor(int16)": np.int16,
    "tensor(uint16)": np.uint16,
    "tensor(int32)": np.int32,
    "tensor(uint32)": np.uint32,
    "tensor(int64)": np.int64,
    "tensor(uint64)": np.uint64,
    "tensor(float16)": np.float16,
    "tensor(float)": np.float32,
    "tensor(double)": np.float64,
    }
    #tensor(int64)
    timestep_dtype = ORT_TO_NP_TYPE[timestep_dtype]

    for i, t in enumerate(tqdm(pipe2.scheduler.timesteps)):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = np.concatenate([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe2.scheduler.scale_model_input(torch.from_numpy(latent_model_input), t)
        latent_model_input = latent_model_input.cpu().numpy()

        # predict the noise residual
        timestep = np.array([t], dtype=timestep_dtype)
        noise_pred = pipe2.unet(sample=latent_model_input, timestep=timestep, encoder_hidden_states=prompt_embeds)
        noise_pred = noise_pred[0]
        guidance_scale = 7.5
        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        scheduler_output = pipe2.scheduler.step(
            torch.from_numpy(noise_pred), t, torch.from_numpy(latents), **extra_step_kwargs
        )
        latents = scheduler_output.prev_sample.numpy()

    latents = 1 / 0.18215 * latents
    ############################################################
    # server 把latents 传输给 手机 client 进行decode 显示生成的图片
    ############################################################
    # image = self.vae_decoder(latent_sample=latents)[0]
    # it seems likes there is a strange result for using half-precision vae decoder if batchsize>1
    image = np.concatenate(
        [pipe.vae_decoder(latent_sample=latents[i : i + 1])[0] for i in range(latents.shape[0])]
    )

    image = np.clip(image / 2 + 0.5, 0, 1)
    image = image.transpose((0, 2, 3, 1))
    image = numpy_to_pil(image)
    image[0].save(f"andriod-generated_image_11111.png")




if __name__ == '__main__':
    main()
