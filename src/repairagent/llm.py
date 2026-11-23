"""Batched chat generation backends.

``HFChat`` wraps a local Hugging Face causal LM (left padding, batched
``generate``). ``ScriptedChat`` replays canned replies and is used in tests.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

Message = dict[str, str]


@dataclass
class Generation:
    text: str
    new_tokens: int


class HFChat:
    def __init__(self, model_id: str, device: str = "cuda", dtype: str = "bfloat16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype)).to(device)
        self.model.eval()
        self.device = device

    def generate(
        self,
        conversations: Sequence[list[Message]],
        max_new_tokens: int = 384,
        temperature: float = 0.0,
        seed: int = 0,
    ) -> list[Generation]:
        torch = self.torch
        prompts = [
            self.tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in conversations
        ]
        enc = self.tok(prompts, return_tensors="pt", padding=True).to(self.device)
        kwargs = {"max_new_tokens": max_new_tokens, "pad_token_id": self.tok.pad_token_id}
        if temperature > 0:
            torch.manual_seed(seed)
            kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
        else:
            kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        with torch.inference_mode():
            out = self.model.generate(**enc, **kwargs)
        new = out[:, enc["input_ids"].shape[1] :]
        res = []
        for row in new:
            n = int((row != self.tok.pad_token_id).sum())
            res.append(Generation(self.tok.decode(row, skip_special_tokens=True), n))
        return res


@dataclass
class ScriptedChat:
    """Deterministic fake: ``reply(conversation)`` decides the answer."""

    reply: Callable[[list[Message]], str]
    calls: list[int] = field(default_factory=list)

    def generate(self, conversations, max_new_tokens=384, temperature=0.0, seed=0):
        self.calls.append(len(conversations))
        return [Generation(self.reply(list(c)), 10) for c in conversations]
