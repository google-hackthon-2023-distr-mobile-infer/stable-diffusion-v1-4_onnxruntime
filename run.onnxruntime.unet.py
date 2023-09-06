import os
from diffusers import OnnxStableDiffusionPipeline, OnnxRuntimeModel
from diffusers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler, DPMSolverMultistepScheduler
from transformers import CLIPTextModel, CLIPTokenizer
 
model_dir = "/Users/zhangyixin/Desktop/hackathon-google/stable-diffusion-v1-4_onnxruntime/"
 
prompt = "a photo of an astronaut riding a horse on mars"
 
num_inference_steps = 20
 
scheduler = PNDMScheduler.from_pretrained(os.path.join(model_dir, "scheduler/scheduler_config.json"))
 
tokenizer = CLIPTokenizer.from_pretrained(model_dir, subfolder="tokenizer")
 
text_encoder = OnnxRuntimeModel(model=OnnxRuntimeModel.load_model(os.path.join(model_dir, "text_encoder/model.onnx")))
 
# in txt to image, vae_encoder is not necessary, only used in image to image generation
# vae_encoder = OnnxRuntimeModel(model=OnnxRuntimeModel.load_model(os.path.join(model_dir, "vae_encoder/model.onnx")))
 
vae_decoder = OnnxRuntimeModel(model=OnnxRuntimeModel.load_model(os.path.join(model_dir, "vae_decoder/model.onnx")))

#放到了3060gpu服务器上？
# 这里循环20次 ，encoder 会参与吗？
unet = OnnxRuntimeModel(model=OnnxRuntimeModel.load_model(os.path.join(model_dir, "unet/model.onnx")))
 
#  image = pipe(prompt, guidance_scale=7.5, num_inference_steps=15, generator=generator).images[0]

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
 
image = pipe(prompt, num_inference_steps=num_inference_steps, height=64, width=64).images[0]
 
image.save(f"generated_image_mac_m1.png")