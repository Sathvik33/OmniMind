import torch
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig


class VisionService:

    def __init__(self, model_name: str = "llava-hf/llava-1.5-7b-hf"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=True
        )

        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map={"": 0}
            
        )

        self.processor = AutoProcessor.from_pretrained(model_name,use_fast=True)

    def describe(self, image_path: str) -> str:
        image = Image.open(image_path).convert("RGB")

        prompt = (
            "USER: <image>\n"
            "Describe what is happening in this image that has to be breif and if any text is present return that text too in two or concise sentences and that has to be accurate.\n"
            "ASSISTANT:"
        )

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=60,
                do_sample=False,
                KV_cache=True
            )

        description = self.processor.batch_decode(
            output,
            skip_special_tokens=True
        )[0]

        return description.split("ASSISTANT:")[-1].strip()