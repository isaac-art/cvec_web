import os
import gc
import time
import uuid
import json
import torch
import sqlite3
from threading import Thread
from datetime import datetime
from pydantic import BaseModel
from contextlib import contextmanager
from huggingface_hub import login, whoami
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from typing import Dict, Any, List, Tuple, Union, Optional, Callable

# try import our local admin file which has the private keys
# if not load the default with the other things
try: from settings_admin import *
except ImportError: from settings import *

from repeng import ControlVector, ControlModel, DatasetEntry

#######################
###      APIS       ###
#######################
os.environ['HF_HOME'] = HF_HOME
login(HF_KEY)
user_info = whoami()
print(f"Logged in as: {user_info['name']} ({user_info['email']})")

#######################
###       DB       ###
#######################

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    try: yield conn
    finally: conn.close()

def init_db():
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS vectors (uuid TEXT PRIMARY KEY, name TEXT, location TEXT, project TEXT, model TEXT, layers TEXT, status TEXT, pos TEXT, neg TEXT, created_at TEXT)")
        conn.commit()

#######################
###     SERVER      ###
#######################
model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

app = FastAPI(title="CVEC API", description="ControlVecAPI", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/static/vectors", StaticFiles(directory="vectors"), name="vectors")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "message": str(exc)}
    )

@app.on_event("startup")
async def startup_event():
    global model, tokenizer, device
    init_db()
    setup = CVector(
        uuid=str(uuid.uuid4()),
        location="none",
        name="default", project="default",
        pos=[], neg=[],
        model=CV_DEFAULT_MODEL,
        layers=CV_DEFAULT_LAYERS,
        created_at=datetime.now().strftime('%Y%m%d')
    )

    ret, model_init = prep_model(setup)
    if not ret: raise model_init
    model, tokenizer, device = model_init

#######################
###    REQ, RES     ###
#######################
class CVectorRequest(BaseModel):
    name: str
    project: str
    pos: List[str]
    neg: List[str]
    model: str = CV_DEFAULT_MODEL
    layers: List[int] = CV_DEFAULT_LAYERS

class CVector(BaseModel):
    uuid: str
    name: str
    location: str
    created_at: str
    pos: List[str]
    neg: List[str]
    status: str = "queued"
    project: str = "default"
    model: str = CV_DEFAULT_MODEL
    layers: List[int] = CV_DEFAULT_LAYERS

class PromptRequest(BaseModel):
    prompt: str
    control_vector_weights: List[Tuple[str, float]]

#######################
###       API       ###
#######################

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/moon")
async def moon():
    return FileResponse("static/moon.html")

@app.get("/axis")
async def axis():
    return FileResponse("static/axis.html")


@app.get("/sys")
async def sys_check():
    return {
        "os": os.name,
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
        "hf_home": HF_HOME,
    }

@app.post("/generate")
async def generate(pr:PromptRequest):
    stream_response = run_generation(pr.control_vector_weights, pr.prompt)
    return StreamingResponse(stream_response, media_type="text/plain")

@app.get("/projects", response_model=List[str])
async def get_projects():
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT project FROM vectors")
        projects = cursor.fetchall()
    return [project[0] for project in projects]

@app.get("/vectors", response_model=List[CVector])
async def get_vectors():
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM vectors")
        vectors = cursor.fetchall()
    return [CVector(**{
        **dict(zip(["uuid", "name", "location", "project", "model", "status", "pos", "neg","created_at"], vector[:5] + vector[6:])),
        "layers": [int(x) for x in vector[5].split(",")],
        "pos": [x for x in vector[7].split(",")],
        "neg": [x for x in vector[8].split(",")],
    }) for vector in vectors]

@app.get("/vectors/{id}", response_model=CVector)
async def get_vector(id: str):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM vectors WHERE id = ?", (id,))
        vector = cursor.fetchone()
    if not vector:
        raise HTTPException(status_code=404, detail="Vector not found")
    return CVector(**{
        **dict(zip(["uuid", "name", "location", "project", "model", "status", "pos", "neg", "created_at"], vector[:5] + vector[6:])),
        "layers": [int(x) for x in vector[5].split(",")],
        "pos": [x for x in vector[7].split(",")],
        "neg": [x for x in vector[8].split(",")],
    })

@app.get("/vectors/project/{project}", response_model=List[CVector])
async def get_vectors_by_project(project: str):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM vectors WHERE project = ?", (project,))
        vectors = cursor.fetchall()
        return [CVector(**{
            **dict(zip(["uuid", "name", "location", "project", "model", "status", "pos", "neg", "created_at"], vector[:5] + vector[6:])),
            "layers": [int(x) for x in vector[5].split(",")],
            "pos": [x for x in vector[7].split(",")],
            "neg": [x for x in vector[8].split(",")],
        }) for vector in vectors]

@app.get("/vector/{uuid}/layers")
async def get_vector_layers(uuid: str):
    # return the numpy lists for the file at location 
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT location FROM vectors WHERE uuid = ?", (uuid,))
        location = cursor.fetchone()[0]
    vector = ControlVector.import_gguf(location)
    print(vector.directions)
    return {
        "layers": {str(layer): direction.tolist() for layer, direction in vector.directions.items()}
    }


@app.delete("/vectors/{id}")
async def delete_vector(id: str):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM vectors WHERE uuid = ?", (id,))
        conn.commit()
    return {"status": "success"}

@app.post("/train")
async def train(cv:CVectorRequest):
    # print(cv)
    filename = f"{cv.name.strip().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.gguf"
    location = f"vectors/{cv.project}/{filename}"

    v = CVector(
        uuid=str(uuid.uuid4()),
        name=cv.name,
        location=location,
        project=cv.project,
        model=cv.model,
        layers=cv.layers,
        status="queued",
        pos=cv.pos,
        neg=cv.neg,
        created_at=datetime.now().strftime('%Y%m%d')
    )
    with get_db() as conn:
        cursor = conn.cursor()
        layers_db = ",".join(str(layer) for layer in v.layers)
        pos_db = ",".join(v.pos)
        neg_db = ",".join(v.neg)
        cursor.execute("INSERT INTO vectors (uuid, name, location, project, model, layers, status, pos, neg, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (v.uuid, v.name, v.location, v.project, v.model, layers_db, v.status, pos_db, neg_db, v.created_at))
        conn.commit()
    return v


@app.get("/reset/{uuid}")
async def reset(uuid: str):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE vectors SET status = 'queued' WHERE uuid = ?", (uuid,))
        conn.commit()
    return {"status": "success"}


@app.get("/demo")
async def demo():
    # Create some demo vectors for testing
    demo_vectors = [
        CVectorRequest(
            name="Sun",
            project="moon",
            pos=["You are the sun", "You are a bright star", "You provide light and warmth"],
            neg=["You are cold", "You are dark", "You are an AI language model"],
        )
    ]

    results = []
    for vector in demo_vectors:
        result = await train(vector)
        results.append(result)
    
    return results

@app.post("/clear_chat")
async def clear_chat():
    global chat_history
    chat_history = []
    return {"status": "success"}

@app.post("/archive_chat")
async def archive_chat():
    global chat_history
    archive_filename = f"chat_archive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(archive_filename, 'w') as archive_file:
        json.dump(chat_history, archive_file)
    return {"status": "success", "archive": archive_filename}

#######################
###    GENERATE     ###
#######################
def load_weighted_vectors(control_vector_weights:List[tuple[str, float]]) -> Tuple[bool, Union[Tuple[ControlVector, CVector], Exception]]:
    try:
        print(f"Loading {len(control_vector_weights)} vectors")
        f_vec = None
        vectors = []
        for vector_uuid, vector_weight in control_vector_weights:
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM vectors WHERE uuid = ?", (vector_uuid,))
                vector = cursor.fetchone()
                if not vector: raise Exception(f"Vector {vector_uuid} not found")
            # load the gguf
            vector = CVector(**dict(zip(["uuid", "name", "location", "project", "model", "status", "created_at"], vector[:5] + vector[6:])), layers=[int(x) for x in vector[5].split(",")], pos=[x for x in vector[7].split(",")], neg=[x for x in vector[8].split(",")])
            if f_vec is None: f_vec = vector
            vector = ControlVector.import_gguf(vector.location)
            vectors.append(vector * vector_weight)
        if len(vectors) == 1:
            return True, (vectors[0], f_vec)
        else:   
            print(f"Summing {len(vectors)} vectors")
            final_vector = vectors[0]  #sum(vectors)
            # ControlVector()
            #     model_type: str
            #     directions: dict[int, np.ndarray]
            for vector in vectors[1:]:
                # check model type is same as final_vector
                if vector.model_type != final_vector.model_type:
                    raise Exception(f"Model type mismatch: {vector.model_type} != {final_vector.model_type}")
                # check layers are same as final_vector
                if vector.directions.keys() != final_vector.directions.keys():
                    raise Exception(f"Layers mismatch: {vector.directions.keys()} != {final_vector.directions.keys()}")
                # sum the vectors
                final_vector.directions = {layer: final_vector.directions[layer] + vector.directions[layer] for layer in final_vector.directions}
            return True, (final_vector, f_vec)
    except Exception as e:
        print(f"Error during load_weighted_vectors: {e}")
        return False, e

def run_generation(control_vector_weights:List[tuple[str, float]], prompt:str):
    global model, tokenizer, device, chat_history
    try:
        res, data = load_weighted_vectors(control_vector_weights)
        if not res: raise data
        final_vector, f_vec = data

        chat_history.append({"role": "user", "content": prompt})
        prompt_input = chat_template_unparse([(msg["role"], msg["content"]) for msg in chat_history])

        if model is None or tokenizer is None or device is None:
            res, data = prep_model(f_vec)
            if not res: raise data
            model, tokenizer, device = data
        
        
        max_new_tokens: int = CV_MAX_NEW_TOKENS
        repetition_penalty: float = CV_REPETITION_PENALTY
        show_baseline: bool = CV_SHOW_BASELINE
        temperature: float = CV_TEMPERATURE

        model_inputs = tokenizer(prompt_input, return_tensors="pt").to(device)
        streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)

        print("Resetting model")
        model.reset()
        print("Setting control vector")
        model.set_control(final_vector)

        settings = {
            "pad_token_id": tokenizer.eos_token_id,  # silence warning
            # "do_sample": False, # temperature=0
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty,
        }
        
        print("Starting generation thread")
        generation_thread = Thread(target=model.generate, kwargs={"streamer": streamer, **model_inputs, **settings})
        generation_thread.start()

        model_output = ""
        for new_text in streamer:
            model_output += new_text
            # print(new_text, end="", flush=True)
            yield new_text
        chat_history.append({"role": "assistant", "content": model_output})
        return model_output
    except Exception as e:
        print(f"Error during run_generation: {e}")
        return e

#######################
###     MODELS      ###
#######################

def prep_model(setup:CVector | dict) -> Tuple[bool, Union[Tuple[ControlModel, AutoTokenizer, str], Exception]]:
    try:
        try:
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error during garbage collection and cuda cleanup prep_model: {e}")
            raise e
            
        model_name = setup.model 
        device = "cuda:0" #if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {device}")
        print("Loading tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = 0
        print("Loading model")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )
        print("Wrapping model")
        wrapped_model = model
        model = ControlModel(wrapped_model, setup.layers) 
        return True, (model, tokenizer, device)
    except Exception as e:
        print(f"Error during prep_model: {e}")
        return False, e

#######################
###     WORKERS     ###
#######################
def training_worker():
    while True:
        print(datetime.now().strftime('%Y%m%d %H:%M:%S'), "Train Vector Worker Running", end="\r")
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM vectors WHERE status IN ('queued', 'training')")
            queue_length = cursor.fetchone()[0]
            cursor.execute("SELECT * FROM vectors WHERE status IN ('queued', 'training') LIMIT 1")
            vector = cursor.fetchone()
        if vector:
            print(" ")
            print(datetime.now().strftime('%Y%m%d %H:%M:%S'), f"Current queue length: {queue_length}")
            vector = CVector(**dict(zip(["uuid", "name", "location", "project", "model", "status", "created_at"], vector[:5] + vector[6:])), layers=[int(x) for x in vector[5].split(",")], pos=[x for x in vector[7].split(",")], neg=[x for x in vector[8].split(",")])
            print(datetime.now().strftime('%Y%m%d %H:%M:%S'), f"Processing vector {vector.uuid}")
            try:
                with get_db() as conn:
                    cursor = conn.cursor()
                    cursor.execute("UPDATE vectors SET status = 'training' WHERE uuid = ?", (vector.uuid,))
                    conn.commit()
                # SETUP MODEL
                res, data = prep_model(vector)
                if not res: raise data
                model, tokenizer, device = data
                # MAKE DATASET
                res, data = generate_dataset(vector, tokenizer, CV_DEFAULT_DATASET)
                if not res: raise data
                dataset = data
                # TRAIN VECTOR ON DATASET
                res, data = train_vector(dataset, model, tokenizer, device)
                if not res: raise data
                control_vector = data
                # SAVE VECTOR
                os.makedirs(os.path.dirname(vector.location), exist_ok=True)
                control_vector.export_gguf(vector.location)
                # UPDATE STATUS
                with get_db() as conn:
                    cursor = conn.cursor()
                    cursor.execute("UPDATE vectors SET status = 'trained' WHERE uuid = ?", (vector.uuid,))
                    conn.commit()
            except Exception as e:
                with get_db() as conn:
                    cursor = conn.cursor()
                    cursor.execute("UPDATE vectors SET status = 'failed' WHERE uuid = ?", (vector.uuid,))
                    conn.commit()
                print(f"Error training vector {vector.uuid}: {e}")
        else:
            time.sleep(10)


#######################
### CV TRAIN UTILS  ###
#######################
def train_vector(dataset:List[DatasetEntry], model:ControlModel, tokenizer:AutoTokenizer, device:str) -> Tuple[bool, Union[ControlVector, Exception]]:
    try:
        model.reset()
        vector = ControlVector.train(
            model, tokenizer, dataset, 
            batch_size=CV_BATCH_SIZE, method=CV_METHOD
        )
        return True, vector
    except Exception as e:
        print(f"Error during train_vector: {e}")
        return False, e

def generate_dataset(vector:CVector, tokenizer:AutoTokenizer, dataset_path:str=CV_DEFAULT_DATASET) -> Tuple[bool, Union[List[DatasetEntry], Exception]]:
    try:
        with open(dataset_path) as f:
            output_suffixes = json.load(f)
            truncated_output_suffixes = [
                tokenizer.convert_tokens_to_string(tokens[:i])
                for tokens in (tokenizer.tokenize(s) for s in output_suffixes)
                for i in range(1, len(tokens))
            ]
        res, dataset = make_dataset(
            chat_template_unparse([("user", "{persona}")]),
            vector.pos,
            vector.neg,
            truncated_output_suffixes,
        )
        if not res: raise dataset
        return True, dataset
    except Exception as e:
        print(f"Error during generate_dataset: {e}")
        return False, e

def make_dataset(template: str, positive_personas: list[str],
    negative_personas: list[str], suffix_list: list[str],) -> Tuple[bool, Union[list[DatasetEntry], Exception]]:
    # Create a dataset of positive and negative examples for training
    try:
        dataset = []
        for suffix in suffix_list:
            for positive_persona, negative_persona in zip(positive_personas, negative_personas):
                positive_template = template.format(persona=positive_persona)
                negative_template = template.format(persona=negative_persona)
                dataset.append(
                    DatasetEntry(
                        positive=f"{positive_template}{suffix}",
                        negative=f"{negative_template}{suffix}",
                    )
                )
        return True, dataset
    except Exception as e:
        print(f"Error during make_dataset: {e}")
        return False, e

#######################
###    CHAT UTILS   ###
#######################
def chat_template_unparse(messages: list[tuple[str, str]]) -> str:
    # Convert chat template (role, content) into a string
    template = []
    for role, content in messages:
        template.append(
            f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        )
    if messages[-1][0] != "assistant":
        # prefill assistant prefix
        template.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(template)


def chat_template_parse(resp: str) -> list[tuple[str, str]]:
    # Parse chat template response into list of (role, content) tuples
    resp = resp.strip().removeprefix("<|begin_of_text|>")
    messages = []
    for part in resp.split("<|start_header_id|>"):
        role_and_content = part.split("<|end_header_id|>")
        if len(role_and_content) == 1:
            role, content = role_and_content[0], ""
        else:
            role, content = role_and_content
        content = content.split("<|eot_id|>")[0]
        messages.append((role.strip(), content.strip()))
    return messages
