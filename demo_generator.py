import os
import json
import math
import time
import random
import dataclasses
import tkinter as tk
from tkinter import scrolledtext, ttk
import colorsys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from repeng import ControlVector, ControlModel

CV_BATCH_SIZE = 32
CV_METHOD = "pca_center"
CV_REPETITION_PENALTY = 1.1
CV_TEMPERATURE = 0.7

N_CONTEXT = 50
# CV_DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
CV_DEFAULT_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
CV_DEFAULT_LAYERS = list(range(5, 22))

VECTORS = [
    "vectors/default/Wind_20250123.gguf",
    "vectors/moon/moon_20241218.gguf",
    "vectors/moon/sun_20241217.gguf",
    "vectors/default/drunk_20250107.gguf",
    "vectors/default/Fish_20250107.gguf",
    "vectors/default/Happy_20250107.gguf",
    "vectors/default/Ice_20250107.gguf"
]
MIN_CVEC, MAX_CVEC = -0.5, 0.9

DEVICE = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

PROMPT = "I am"

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
        
        # Load all vectors but only use one at a time
        print("Loading vectors:")
        self.vectors = {}
        for v in VECTORS:
            name = os.path.splitext(os.path.basename(v))[0]
            print(f"Loading: {name}")
            self.vectors[name] = ControlVector.import_gguf(v)

        self.tokens: list[str] = self.tokenizer.tokenize(PROMPT)
        self.step = 0
        self.previous_cvec_applied = None

    def next(self, vector_name: str, raw_strength: float):
        print(f"\nApplying {vector_name} with strength: {raw_strength:.2f}")
        
        strength = (raw_strength + 1) / 2 * (MAX_CVEC - MIN_CVEC) + MIN_CVEC
        vector = self.vectors[vector_name] * strength

        if self.previous_cvec_applied is None or vector != self.previous_cvec_applied:
            self.model.set_control(vector)
            self.previous_cvec_applied = vector

        context = self.tokenizer.convert_tokens_to_string(self.tokens[-N_CONTEXT:])
        model_tokens = self.tokenizer(context, return_tensors="pt").to(self.model.device)
        logits = self.model.forward(**model_tokens).logits[0, -1, :]
        logits[self.tokenizer.eos_token_id] = -10000
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        self.tokens.append(self.tokenizer.decode(next_token))
        self.step += 1

        return Token(
            content=self.tokens[-1],
            strength=strength,
        )

class TextDisplay:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Generated Text Display")
        self.root.geometry("800x600")
        
        # Create control frame
        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(fill='x', padx=10, pady=5)
        
        # Add vector selector dropdown
        vector_frame = tk.Frame(self.control_frame)
        vector_frame.pack(fill='x', pady=2)
        tk.Label(vector_frame, text="Vector:", width=10, anchor='e').pack(side='left')
        
        self.vector_names = [os.path.splitext(os.path.basename(v))[0] for v in VECTORS]
        self.selected_vector = tk.StringVar(value=self.vector_names[0])
        vector_dropdown = ttk.Combobox(
            vector_frame, 
            textvariable=self.selected_vector,
            values=self.vector_names,
            state='readonly'
        )
        vector_dropdown.pack(side='left', fill='x', expand=True)
        
        # Add strength slider
        slider_frame = tk.Frame(self.control_frame)
        slider_frame.pack(fill='x', pady=2)
        tk.Label(slider_frame, text="Strength:", width=10, anchor='e').pack(side='left')
        
        self.strength_var = tk.DoubleVar(value=0.0)
        self.strength_slider = tk.Scale(
            slider_frame,
            from_=-1.0,
            to=1.0,
            resolution=0.01,
            orient='horizontal',
            variable=self.strength_var,
            length=300
        )
        self.strength_slider.pack(side='left', fill='x', expand=True)
        
        # Create text area
        self.text_area = scrolledtext.ScrolledText(
            self.root, 
            wrap=tk.WORD,
            width=80,
            height=20,
            font=("Courier", 12)
        )
        self.text_area.pack(expand=True, fill='both', padx=10, pady=10)
    
    def get_vector_and_strength(self) -> tuple[str, float]:
        return self.selected_vector.get(), self.strength_var.get()
    
    def strength_to_color(self, strength: float) -> str:
        normalized = (strength - MIN_CVEC) / (MAX_CVEC - MIN_CVEC)
        hue = (1 - normalized) * 120 / 360
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        return f'#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}'
    
    def add_token(self, token: Token):
        self.text_area.tag_config(
            f"strength_{self.text_area.index('end-1c')}", 
            foreground=self.strength_to_color(token.strength)
        )
        self.text_area.insert(
            'end',
            token.content, 
            f"strength_{self.text_area.index('end-1c')}"
        )
        self.text_area.see('end')
        self.root.update()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    generator = Generator()
    display = TextDisplay()
    
    def generate():
        vector_name, strength = display.get_vector_and_strength()
        token = generator.next(vector_name, strength)
        display.add_token(token)
        display.root.after(50, generate)
    
    display.root.after(100, generate)
    display.run()