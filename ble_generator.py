import os
import json
import math
import time
import random
import datetime
import dataclasses
import asyncio
import struct
from bleak import BleakClient
import colorsys
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
import uvicorn

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from repeng import ControlVector, ControlModel

CV_BATCH_SIZE = 32
CV_METHOD = "pca_center"
CV_REPETITION_PENALTY = 1.1
CV_TEMPERATURE = 0.7

N_CONTEXT = 500
CV_DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
CV_DEFAULT_LAYERS = list(range(5, 22))

CVEC = "vectors/moon/moon_20241218.gguf"
MIN_CVEC, MAX_CVEC = -0.8, 1.2

DEVICE = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

PROMPT = "Describe what you see: "

# BLE constants
IMU_SERVICE_UUID = "eb1d3224-ab67-4114-89db-d12ac0684005"
IMU_DATA_UUID = "963eeca0-d121-458c-b32f-a99c40d8bf19"
DEVICE_ADDRESS = "753E1AA1-3AD1-DEF4-5B4A-CF09F9640206"

# Global state
current_strength = 0.0
generation_active = True
token_queue = asyncio.Queue()
start_time = time.time()  # Add this line to track time for sine wave

sinwave_mode = True  # This existing line will control which mode we use

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def get_page():
    return FileResponse("static/stream.html")

@app.get("/stream")
async def stream_text(request: Request):
    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                
                # Get next token from queue
                token = await token_queue.get()
                yield {
                    "data": json.dumps({
                        "content": token.content,
                        "strength": token.strength
                    })
                }
        except asyncio.CancelledError:
            pass
    
    return EventSourceResponse(event_generator())

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
        # Map yaw to smooth circular pattern: -180/180° -> 0, 90° -> 1, -90° -> -1
        normalized_yaw = math.sin(math.radians(yaw))  # Convert to radians and apply sine
        current_strength = normalized_yaw
        # print(f"Current strength: {current_strength:.2f}")

def get_sine_strength() -> float:
    """Calculate sine wave strength based on current second in minute"""
    current_second = datetime.datetime.now().second
    # Map seconds (0-59) to radians (0-2π) and shift by π/2 to start at -1
    angle = (current_second / 60) * 2 * math.pi - (math.pi / 2)
    return math.sin(angle)

async def run_ble():
    global generation_active, current_strength
    ble = BLEController()
    try:
        if sinwave_mode:
            while generation_active:
                current_strength = get_sine_strength()
                await asyncio.sleep(0.1)  # Update every 100ms
        else:
            print(f"Connecting to BLE device at {DEVICE_ADDRESS}...")
            async with BleakClient(DEVICE_ADDRESS) as client:
                print("Connected! Reading orientation data...")
                
                ble.client = client
                await client.start_notify(IMU_DATA_UUID, ble.notification_handler)
                
                while generation_active:
                    await asyncio.sleep(0.1)

    except Exception as e:
        print(f"\nBLE Error: {str(e)}")
        generation_active = False

async def run_generator():
    global generation_active
    generator = Generator()
    # print(PROMPT, end='', flush=True)
    
    # Put initial prompt in queue
    await token_queue.put(Token(content=PROMPT, strength=0))
    
    try:
        while generation_active:
            token = generator.next(current_strength)
            await token_queue.put(token)
            await asyncio.sleep(0.001)
            
    except Exception as e:
        print(f"\nGenerator Error: {str(e)}")
        generation_active = False

async def run_server():
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

async def main():
    global generation_active
    try:
        # Run all processes concurrently
        await asyncio.gather(
            run_ble(),
            run_generator(),
            run_server()
        )
    except KeyboardInterrupt:
        print("\nStopping...")
        generation_active = False
    finally:
        generation_active = False

if __name__ == "__main__":
    asyncio.run(main())