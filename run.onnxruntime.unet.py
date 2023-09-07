import argparse
from diffusers import OnnxStableDiffusionPipeline, OnnxRuntimeModel
from diffusers.pipelines.onnx_utils import ORT_TO_NP_TYPE
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

import inspect
import time

import os

SD_CPU_ROOT=os.path.join(os.path.dirname(__file__), ".")
 
def logging(msg):
    logger.info(msg) 

def load_onnx(subgraph_name, model_dir):
    logging(f"loading {subgraph_name}...")
    start = time.time()
    # equivalent to `onnx.load_model(str_or_path)`
    model = OnnxRuntimeModel.load_model(os.path.join(model_dir, subgraph_name))
    elapse = time.time() - start
    logging(f"{subgraph_name} loaded, elapse {elapse:.3f} s")
    return model

# very fast
def tokenize_and_encode_text(subgraph, inputs, full_pipe):
    # TODO (yiakwy) check inputs
    prompt, num_inference_steps, height, width = inputs
    
    # see OnnxStableDiffusionPipeline.__call__
    full_pipe.check_inputs(prompt, height, width, 1, None, None, None)
    
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        raise Exception("Cannot decide batch size")
    # (2, 77, 768)
    prompt_embeds = full_pipe._encode_prompt(prompt, 1, True, None, None, None)
    return prompt_embeds, batch_size

# very slow
def unet_sampling_loop(subgraph, inputs, full_pipe):
    # TODO (yiakwy) check inputs
    prompt_embeds, batch_size, num_inference_steps, height, width = inputs
    
    # see OnnxStableDiffusionPipeline.__call__
    # get the initial random noise unless the user supplied it
    latents_dtype = prompt_embeds.dtype
    latents_shape = (batch_size * 1, 4, height // 8, width // 8)
    latents = np.random.randn(*latents_shape).astype(latents_dtype)
    
    # set timesteps
    full_pipe.scheduler.set_timesteps(num_inference_steps)
    
    latents = latents * np.float64(full_pipe.scheduler.init_noise_sigma)
    
    accepts_eta = "eta" in set(inspect.signature(full_pipe.scheduler.step).parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs["eta"] = 0.0
        
    timestep_dtype = next(
        (input.type for input in subgraph.get_inputs() if input.name == "timestep"), "tensor(float)"
    )
    timestep_dtype = ORT_TO_NP_TYPE[timestep_dtype]
    
    for i, t in enumerate(full_pipe.progress_bar(full_pipe.scheduler.timesteps)):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = np.concatenate([latents] * 2)
        latent_model_input = full_pipe.scheduler.scale_model_input(torch.from_numpy(latent_model_input), t)
        latent_model_input = latent_model_input.cpu().numpy()

        # predict the noise residual
        timestep = np.array([t], dtype=timestep_dtype)
        noise_pred = full_pipe.unet(sample=latent_model_input, timestep=timestep, encoder_hidden_states=prompt_embeds)
        noise_pred = noise_pred[0]
        guidance_scale = 7.5
        # perform guidance
        noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        scheduler_output = full_pipe.scheduler.step(
            torch.from_numpy(noise_pred), t, torch.from_numpy(latents), **extra_step_kwargs
        )
        latents = scheduler_output.prev_sample.numpy()
    # (1, 4, 64, 64)
    latents = 1 / 0.18215 * latents
    return latents
 
# very fast    
def decode_latents(subgraph, inputs, full_pipe):
    # TODO (yiakwy) check inputs
    latents = inputs[0]
    
    image = np.concatenate(
        [full_pipe.vae_decoder(latent_sample=latents[i : i + 1])[0] for i in range(latents.shape[0])]
    )
    
    image = np.clip(image / 2 + 0.5, 0, 1)
    image = image.transpose((0, 2, 3, 1))
    image = numpy_to_pil(image)

    return [image]



def dispatch(stage, subgraphs, inputs, full_pipe):
    subgraph = subgraphs.get(stage, None)
    if subgraph is None:
        raise Exception(f"the requested stage is not found in records")
    
    outputs = []
    if stage == "stage#0":
        if inputs == []:
            # prompt, num_inference_steps, height, width
            inputs = ["a photo of an astronaut riding a horse on mars", 20, 512, 512]
            
        outputs =  tokenize_and_encode_text(subgraph, inputs, full_pipe)
    elif stage == "stage#1":
        if inputs == []:
            # prompt_embeds, batch_size, num_inference_steps, height, width
            prompt_embeds = np.load("prompt_embeds.npy")
            inputs = [prompt_embeds, 1, 20, 512, 512]
        outputs = unet_sampling_loop(subgraph, inputs, full_pipe)
    else:
        if inputs == []:
            latents = np.load("latents.npy")
            inputs = [latents]
        outputs = decode_latents(subgraph, inputs, full_pipe)        
    
    # check outputs
    # TODO (yiakwy)
    
    return outputs
    
    
def main(args):
    model_dir = args.model_dir
    
    ########################################################################################################################
    # step 1 : constuct subgraph with onnx runtime (CPU) backend
    # TODO (yiakwy) remove subgraph not used
    stages = {
        "stage#0" : load_onnx("text_encoder/model.onnx", model_dir) if "stage#0" == args.stage else None,
        "stage#2" : load_onnx("vae_decoder/model.onnx",  model_dir) if "stage#2" == args.stage else None,
        "stage#1" : load_onnx("unet/model.onnx",         model_dir) if "stage#1" == args.stage else None
    } 
    
    scheduler = PNDMScheduler.from_pretrained(os.path.join(model_dir, "scheduler/scheduler_config.json"))
    
    tokenizer = CLIPTokenizer.from_pretrained(model_dir, subfolder="tokenizer")
    
    text_encoder = OnnxRuntimeModel(model=stages["stage#0"]) if stages["stage#0"] != None else None
    
    vae_decoder  = OnnxRuntimeModel(model=stages["stage#2"]) if stages["stage#2"] != None else None

    unet         = OnnxRuntimeModel(model=stages["stage#1"]) if stages["stage#1"] != None else None
    
    ########################################################################################################################
    # step 2 : constuct subgraph with onnx runtime (CPU) backend
    # Trick : constuct full pipeline graph
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
    
    # execute subgraph and execute with inputs on CPU (ORT session)
    outputs = dispatch(args.stage, stages, args.inputs, pipe)
    
    print(outputs)    
    return outputs


def parse_args():
    parser = argparse.ArgumentParser(
        description='end 2 end demo'
    )
    parser.add_argument('--model_dir', 
                        type=str, 
                        default=f"{SD_CPU_ROOT}/stable-diffusion-v1-4/", 
                        help="Full path of the onnx model.")
    parser.add_argument('--text2img',
                        type=bool,
                        default=True,
                        help="text2img or img2img")
    parser.add_argument('--stage',
                        type=str,
                        default='stage#0',
                        help="stage index for this endpoint")
    parser.add_argument('--inputs',
                        type=list,
                        default=[],
                        help="assumed to backup from numpy bytearray")
    parser.add_argument('--outputs',
                        type=list,
                        default=[],
                        help="assumed to backup from numpy bytearray")
    
    args = parser.parse_args()
    return args, parser

if __name__ == '__main__':
    args, _ = parse_args()
    
    main(args)
