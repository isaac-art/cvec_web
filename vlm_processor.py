from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import cv2
import torch
from PIL import Image
import io

class VLM:
    def __init__(self, use_flash_attention: bool = False, device: str = "mps"):
        """Initialize the VLM processor with optional flash attention."""
        print("Initializing VLM")
        model_name = "Qwen/Qwen2-VL-2B-Instruct"
        
        if use_flash_attention:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto"
            )
        else:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto"
            )
        
        self.model.to(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        print("VLM initialized")
        
    def capture_image(self) -> Image.Image:
        """Capture an image from the webcam and return as PIL Image."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened(): raise RuntimeError("Could not access webcam")
        # Wait for camera to initialize and capture non-black frames
        max_attempts = 10
        for _ in range(max_attempts):
            ret, frame = cap.read()
            if not ret: continue
            # Check if image is mostly black
            # Average pixel value threshold
            if cv2.mean(frame)[0] < 5: continue
            # Valid frame captured
            cap.release()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb_frame)
        # Release camera if no valid frame captured
        cap.release()
        raise RuntimeError("Failed to capture non-black image after multiple attempts")

    def get_image_description(self, image: Image.Image) -> str:
        """Get description for a given PIL Image."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]

        # Prepare for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # Generate response
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        
        return output_text[0]

    def get_image_and_description(self) -> str:
        """Capture an image from webcam and get its description."""
        try:
            print("Capturing image...")
            image = self.capture_image()
            print("Getting image description...")
            desc = self.get_image_description(image)
            print(desc)
            return desc
        except Exception as e:
            return f"Error processing image: {str(e)}" 
        

if __name__ == "__main__":
    vlm = VLM()
    print(vlm.get_image_and_description())
