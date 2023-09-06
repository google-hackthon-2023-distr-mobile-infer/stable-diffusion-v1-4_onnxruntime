# stable-diffusion-v1-4_onnxruntime
pip install diffusers==0.10.2 transformers scipy ftfy accelerate

pip install onnxruntime


git clone https://huggingface.co/CompVis/stable-diffusion-v1-4 -b onnx


stable-diffusion-v1-4 


model_dir = "/Users/zhangyixin/Desktop/sdlv14/stable-diffusion-v1-4/"


image = pipe(prompt, num_inference_steps=num_inference_steps, height=64, width=64).images[0]

