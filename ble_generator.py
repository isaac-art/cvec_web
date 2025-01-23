import os
import json
import math
import time
import random
import dataclasses
import asyncio
import struct
from bleak import BleakClient
import colorsys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from repeng import ControlVector, ControlModel

CV_BATCH_SIZE = 32
CV_METHOD = "pca_center"
CV_REPETITION_PENALTY = 1.1
CV_TEMPERATURE = 0.7

N_CONTEXT = 50
CV_DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
# CV_DEFAULT_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
CV_DEFAULT_LAYERS = list(range(5, 22))

CVEC = "vectors/default/Wind_20250123.gguf"
MIN_CVEC, MAX_CVEC = -0.5, 0.9

DEVICE = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

PROMPT = "I am"

# BLE constants
IMU_SERVICE_UUID = "eb1d3224-ab67-4114-89db-d12ac0684005"
IMU_DATA_UUID = "963eeca0-d121-458c-b32f-a99c40d8bf19"
DEVICE_ADDRESS = "753E1AA1-3AD1-DEF4-5B4A-CF09F9640206"

# Global state
current_strength = 0.0
generation_active = True

@dataclasses.dataclass
class Token:
    content: str
    strength: float

class Generator:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(CV_DEFAULT_MODEL)
        self.tokenizer.pad_token_id = 0
        model = AutoModelForCausalLM.from_pretrained(CV_DEFAULT_MODEL, torch_dtype=torch.float16).to(DEVICE)
        self.model = ControlModel(model, CV_DEFAULT_LAYERS)
        
        print("Loading vector...")
        self.vector = ControlVector.import_gguf(CVEC)
        self.tokens: list[str] = self.tokenizer.tokenize(PROMPT)
        self.step = 0
        self.previous_cvec_applied = None

    def next(self, raw_strength: float):
        strength = (raw_strength + 1) / 2 * (MAX_CVEC - MIN_CVEC) + MIN_CVEC
        vector = self.vector * strength

        if self.previous_cvec_applied is None or vector != self.previous_cvec_applied:
            # print(f"\nApplying strength: {strength:.2f}")
            self.model.set_control(vector)
            self.previous_cvec_applied = vector

        context = self.tokenizer.convert_tokens_to_string(self.tokens[-N_CONTEXT:])
        model_tokens = self.tokenizer(context, return_tensors="pt").to(self.model.device)
        logits = self.model.forward(**model_tokens).logits[0, -1, :]
        logits[self.tokenizer.eos_token_id] = -10000
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        token_text = self.tokenizer.decode(next_token)
        self.tokens.append(token_text)
        self.step += 1

        return Token(content=token_text, strength=strength)

class BLEController:
    def __init__(self):
        self.client = None
        
    def parse_imu_data(self, data: bytearray) -> tuple:
        """Parse the raw IMU data bytes into orientation values."""
        values = struct.unpack('3f', data)  # 3 float values: pitch, roll, yaw
        return values
        
    def notification_handler(self, sender, data):
        """Handle incoming notifications from the BLE device."""
        global current_strength
        pitch, roll, yaw = self.parse_imu_data(data)
        # Map yaw to [-1, 1]
        normalized_yaw = yaw / 180.0  # Assuming yaw is in degrees [-180, 180]
        normalized_yaw = max(-1.0, min(1.0, normalized_yaw))  # Clamp to [-1, 1]
        current_strength = normalized_yaw

async def run_ble():
    global generation_active  # Declare global at start of function
    ble = BLEController()
    try:
        print(f"Connecting to BLE device at {DEVICE_ADDRESS}...")
        async with BleakClient(DEVICE_ADDRESS) as client:
            print("Connected! Reading orientation data...")
            
            ble.client = client
            await client.start_notify(IMU_DATA_UUID, ble.notification_handler)
            
            while generation_active:
                await asyncio.sleep(0.01)  # Fast updates for smooth control

    except Exception as e:
        print(f"\nBLE Error: {str(e)}")
        generation_active = False

async def run_generator():
    global generation_active  # Declare global at start of function
    generator = Generator()
    print(PROMPT, end='', flush=True)
    
    try:
        while generation_active:
            token = generator.next(current_strength)
            print(token.content, end='', flush=True)
            await asyncio.sleep(0.05)  # Control generation speed
            
    except Exception as e:
        print(f"\nGenerator Error: {str(e)}")
        generation_active = False

async def main():
    global generation_active  # Declare global at start of function
    try:
        # Run both processes concurrently
        await asyncio.gather(
            run_ble(),
            run_generator()
        )
    except KeyboardInterrupt:
        print("\nStopping...")
        generation_active = False
    finally:
        generation_active = False  # Ensure cleanup on any exit

if __name__ == "__main__":
    asyncio.run(main())