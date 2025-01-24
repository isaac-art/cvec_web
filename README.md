# CONTROL VECTOR TRAINING WEB INTERFACE

## SETUP


NOTE: if on runpod remember to set HF_HOME to workspace cache (and HF_TOKEN too)

WARN: on run it will download "meta-llama/Meta-Llama-3-8B-Instruct" if its not present in current HF_HOME


```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

I include `https://github.com/vgel/repeng/` in this repo for the `repeng` library

## ARCHITECTURE

The project consists of two main components: a FastAPI web server and a training worker process. The web interface allows users to create, manage, and use control vectors, while the worker handles the training queue.

### System Architecture

```mermaid
graph TD
    A[Web Interface] -->|HTTP| B[FastAPI Server]
    B -->|Read/Write| C[(SQLite DB)]
    B -->|Load/Save| D[Vector Files]
    E[Training Worker] -->|Process Queue| C
    E -->|Train| F[LLM Model]
    E -->|Save| D
    B -->|Generate| F
    G[BLE Generator] -->|Stream| H[Browser SSE]
```

### Training Workflow

```mermaid
sequenceDiagram
    participant U as User
    participant S as Server
    participant DB as SQLite
    participant W as Worker
    participant M as LLM Model

    U->>S: Create Vector Request
    S->>DB: Queue Vector (status: queued)
    W->>DB: Poll for queued vectors
    W->>DB: Update status (training)
    W->>M: Load model
    W->>M: Generate dataset
    W->>M: Train vector
    W->>S: Save vector file
    W->>DB: Update status (trained)
    U->>S: Use Vector
    S->>M: Load & apply vector
```

## RUN SERVER
start the server and show the web interface

```bash
uvicorn main:app --reload
```

visit `http://localhost:8000` or `http://localhost:8000/docs`
can load and test vectors or submit new ones to the training queue

## RUN WORKER

```bash
python run_worker.py
```
worker will process the queue and train new vectors


### SCREENSHOTS
![Generate](screen_shots/generate.png)

---

![List](screen_shots/list.png)

---

![Create Vector](screen_shots/create.png)

---

![View](screen_shots/list.png)


---

![Live Stream - editing vector strengthper token](screen_shots/livestream.png)