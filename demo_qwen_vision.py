from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import cv2
import time
from pathlib import Path

from repeng import ControlModel, ControlVector

def setup_model():
    device = "mps" if torch.backends.mps.is_available() else "cuda"
    print("Using device:", device)
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    
    # model = ControlModel(model, layer_ids=list(range(5, 22)))
    # model.reset()
    # vector = ControlVector.import_gguf("vectors/qwenv_cvec.gguf")
    # model.set_control(vector*0.9)
    return model, processor, device

def capture_and_save_frame():
    # Create temp directory if it doesn't exist
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera")
    
    # Capture frame
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret: 
            raise RuntimeError("Failed to capture frame")
        i += 1
        if i > 100:break
    
    # Save frame
    temp_path = temp_dir / "current_frame.jpg"
    # resize to 50%
    frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    cv2.imwrite(str(temp_path), frame)
    cv2.imshow("frame", frame)
    cv2.waitKey(0)
    # Release camera
    cap.release()
    cv2.destroyAllWindows()
    return str(temp_path)

def analyze_image(image_path, model, processor, device):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": "What do you see"},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    generated_ids = model.generate(**inputs) #, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

def main():
    print("Initializing model...")
    model, processor, device = setup_model()
    
    try:
        while True:
            input("Press Enter to capture and analyze an image (Ctrl+C to exit)...")
            
            print("Capturing image...")
            image_path = capture_and_save_frame()
            
            print("Analyzing image...")
            result = analyze_image(image_path, model, processor, device)
            
            print("\nAnalysis Result:")
            print(result)
            print("\n" + "-"*50 + "\n")
            
            # Small delay to prevent rapid captures
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nExiting program...")
    finally:
        # Cleanup temp directory
        import shutil
        shutil.rmtree("temp", ignore_errors=True)

if __name__ == "__main__":
    main()