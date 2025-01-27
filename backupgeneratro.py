
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