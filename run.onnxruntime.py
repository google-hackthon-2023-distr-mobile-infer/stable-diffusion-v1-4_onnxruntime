import argparse
import time

from diffusers import OnnxStableDiffusionPipeline, OnnxRuntimeModel
from diffusers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler, DPMSolverMultistepScheduler
from transformers import CLIPTextModel, CLIPTokenizer

import os

SD_CPU_ROOT=os.path.join(os.path.dirname(__file__), ".")

from diffusers.utils import deprecate, logging
logger = logging.get_logger(__name__)

def logging(msg):
    logger.info(msg)

def main(args):
    model_dir = args.model_dir
    
    num_inference_steps = args.num_inference_steps
    
    prompt = args.prompt
    
    scheduler = PNDMScheduler.from_pretrained(os.path.join(model_dir, "scheduler/scheduler_config.json"))
    
    # construct model parts
    # step #1
    logging("tokenizing ...")
    start = time.time()
    tokenizer = CLIPTokenizer.from_pretrained(model_dir, subfolder="tokenizer")
    logging(f"tokenized, elapse {time.time() - start:.3f} s")
    
    # step #2
    logging("loading text encoder ... ")
    start = time.time()
    text_encoder = OnnxRuntimeModel(model=OnnxRuntimeModel.load_model(os.path.join(model_dir, "text_encoder/model.onnx")))
    logging(f"text encoder loaded, elapse {time.time() - start:.3f} s")
    
    # step #3 : txt to image, vae_encoder is not necessary, only used in image to image generation
    vae_encoder = None
    if not args.text2img:
        vae_encoder = OnnxRuntimeModel(model=OnnxRuntimeModel.load_model(os.path.join(model_dir, "vae_encoder/model.onnx")))
       
    logging("loading vae decoder ...") 
    start = time.time()
    vae_decoder = OnnxRuntimeModel(model=OnnxRuntimeModel.load_model(os.path.join(model_dir, "vae_decoder/model.onnx")))
    logging(f"vae decoder loaded, elapse {time.time() - start:.3f} s")
    

    # step #4
    logging("loading unet ... ")
    start = time.time()
    unet = OnnxRuntimeModel(model=OnnxRuntimeModel.load_model(os.path.join(model_dir, "unet/model.onnx")))
    logging(f"unet loaded, elapse {time.time() - start:.3f} s")
    
    # construct pipeline
    pipe = OnnxStableDiffusionPipeline(
        vae_encoder=vae_encoder,
        vae_decoder=vae_decoder,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    
    # TODO warm up device
    
    # do inference on CPU devcie
    logging("do inference:")
    start = time.time()
    image = pipe(prompt, num_inference_steps=num_inference_steps, height=64, width=64).images[0]
    elapse_s = time.time() - start
    
    # TODO average over iterations
    latency_ms = elapse_s * 1000
    print(f"Latency : {latency_ms} ms")
    
    image.save(f"generated_image_mac_m1.png")

def parse_args():
    parser = argparse.ArgumentParser(
        description='end 2 end demo'
    )
    parser.add_argument('--model_dir', 
                        type=str, 
                        default=f"{SD_CPU_ROOT}/stable-diffusion-v1-4/", 
                        help="Full path of the onnx model.")
    parser.add_argument('--num_inference_steps', 
                        type=int, 
                        default=20,
                        help="number of on-chip unet loop count")
    parser.add_argument('--prompt',
                        type=str,
                        default="a photo of an astronaut riding a horse on mars",
                        help="prompt input")
    parser.add_argument('--text2img',
                        type=bool,
                        default=True,
                        help="text2img or img2img")
    
    args = parser.parse_args()
    return args, parser

if __name__ == "__main__":
    args, _ = parse_args()
    
    main(args)