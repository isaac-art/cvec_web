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
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from sse_starlette.sse import EventSourceResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from repeng import ControlVector, ControlModel

CV_BATCH_SIZE = 32
CV_METHOD = "pca_center"
CV_REPETITION_PENALTY = 1.1
CV_TEMPERATURE = 0.7

N_CONTEXT = 5000
CV_DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
# CV_DEFAULT_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
CV_DEFAULT_LAYERS = list(range(5, 22))

CVEC = "vectors/default/Fish_20250107.gguf"
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
token_queue = asyncio.Queue()

app = FastAPI()

# HTML template with JavaScript for SSE handling
HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>Text Generation Viewer</title>
    <style>
        body {
            font-family: monospace;
            margin: 20px;
            transition: background-color 0.3s;
        }
        #text-container {
            font-size: 16px;
            line-height: 1.5;
            white-space: pre-wrap;
        }
        .token {
            transition: color 0.3s;
        }
    </style>
</head>
<body>
    <div id="text-container"></div>
    <script>
        const textContainer = document.getElementById('text-container');
        
        function strengthToColor(strength) {
            // Convert strength [-0.5, 0.9] to hue [120, 0]
            const normalized = (strength - (-0.5)) / (0.9 - (-0.5));
            const hue = (1 - normalized) * 120;
            return `hsl(${hue}, 80%, 40%)`;
        }
        
        function strengthToBackground(strength) {
            const normalized = (strength - (-0.5)) / (0.9 - (-0.5));
            const hue = (1 - normalized) * 120;
            return `hsl(${hue}, 30%, 95%)`;
        }

        const evtSource = new EventSource("/stream");
        evtSource.onmessage = function(event) {
            const data = JSON.parse(event.data);
            
            // Update background color based on strength
            document.body.style.backgroundColor = strengthToBackground(data.strength);
            
            // Add new token with color based on strength
            const span = document.createElement('span');
            span.textContent = data.content;
            span.className = 'token';
            span.style.color = strengthToColor(data.strength);
            textContainer.appendChild(span);
            
            // Scroll to bottom
            window.scrollTo(0, document.body.scrollHeight);
        };
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def get_page():
    return HTML_CONTENT

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
                await asyncio.sleep(0.1)  # Fast updates for smooth control

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