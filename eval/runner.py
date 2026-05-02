from __future__ import annotations

import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

_TRUNC_PATTERNS = (
    "\ndef ",
    "\nclass ",
    "\nif __name__",
    "\nprint(",
    "\nassert ",
    "\n#",
)


def _load_tokenizer(model_path: str):
    # transformers 5.x picks the wrong pre_tokenizer for some BPE configs
    # (DeepSeek-Coder ships a tokenizer.json that gets misread as sentencepiece).
    # round-trip a sentinel; if it loses whitespace, load tokenizer.json raw.
    tok = AutoTokenizer.from_pretrained(model_path)
    sentinel = "def f(x):\n    return x"
    if tok.decode(tok.encode(sentinel, add_special_tokens=False)) == sentinel:
        return tok
    json_path = os.path.join(model_path, "tokenizer.json")
    if not os.path.isfile(json_path):
        return tok
    from tokenizers import Tokenizer
    raw = Tokenizer.from_file(json_path)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        bos_token=getattr(tok, "bos_token", None),
        eos_token=getattr(tok, "eos_token", None),
        pad_token=getattr(tok, "pad_token", None) or getattr(tok, "eos_token", None),
        unk_token=getattr(tok, "unk_token", None),
    )
    return fast


class Generator:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        max_new_tokens: int = 384,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.tokenizer = _load_tokenizer(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=dtype
        ).to(device).eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def complete(self, prompt: str, temperature: float = 0.0, seed: int | None = None) -> str:
        ids = self.tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(self.device)
        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if temperature > 0:
            if seed is not None:
                torch.manual_seed(seed)
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
        else:
            gen_kwargs["do_sample"] = False
        with torch.no_grad():
            out = self.model.generate(**ids, **gen_kwargs)
        gen = out[0, ids["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen, skip_special_tokens=True)
        return _truncate(text)

    def fim_complete(self, prefix: str, suffix: str = "\n", filename: str = "solution.py") -> str:
        wrapped = self._fim_wrap(prefix, suffix, filename)
        if wrapped is None:
            raise RuntimeError(f"model {self.model_path} has no FIM tokens this code knows about")
        ids = self.tokenizer(wrapped, return_tensors="pt", return_token_type_ids=False).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        gen = out[0, ids["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen, skip_special_tokens=False)
        for marker in self._fim_stop_markers():
            i = text.find(marker)
            if i >= 0:
                text = text[:i]
                break
        return _truncate(text)

    def _fim_wrap(self, prefix: str, suffix: str, filename: str) -> str | None:
        vocab = self.tokenizer.get_vocab()
        # Mellum: <fim_prefix>/<fim_suffix>/<fim_middle> + optional <filename>
        if "<fim_middle>" in vocab and "<fim_prefix>" in vocab:
            head = f"<filename>{filename}\n" if "<filename>" in vocab else ""
            return f"{head}<fim_suffix>{suffix}<fim_prefix>{prefix}<fim_middle>"
        # DeepSeek-Coder: full-width pipes around fim▁begin/hole/end
        ds_begin = "<｜fim▁begin｜>"
        ds_hole = "<｜fim▁hole｜>"
        ds_end = "<｜fim▁end｜>"
        if ds_begin in vocab and ds_hole in vocab and ds_end in vocab:
            return f"{ds_begin}{prefix}{ds_hole}{suffix}{ds_end}"
        return None

    def _fim_stop_markers(self) -> tuple[str, ...]:
        return (
            "<filename>", "<fim_suffix>", "<fim_prefix>", "<fim_middle>",
            "<｜fim▁begin｜>", "<｜fim▁hole｜>", "<｜fim▁end｜>",
            "<|endoftext|>", "<｜end▁of▁sentence｜>",
        )


def _truncate(text: str) -> str:
    cuts = [text.find(p) for p in _TRUNC_PATTERNS]
    cuts = [c for c in cuts if c >= 0]
    if cuts:
        text = text[: min(cuts)]
    return text.rstrip() + "\n"


def build_retry_prompt(
    prompt: str,
    prev_completion: str,
    failure: str,
    fmt: str = "current",
) -> str:
    short_failure = re.sub(r"\s+", " ", failure).strip()[:160]
    if fmt == "minimal":
        return f"# Previous attempt was wrong. Try again.\n\n{prompt}"
    if fmt == "traceback":
        return f"# Failed test: {short_failure}\n\n{prompt}"
    if fmt == "post":
        body = "\n".join("    " + line for line in prev_completion.strip().splitlines()[:8])
        return (
            f"{prompt}"
            f"# Previous attempt failed: {short_failure}\n"
            f"# def solution_v1(...):\n"
            f"{body}\n\n"
            f"# Try again with a corrected version below.\n"
        )
    indented = "\n".join("# " + line for line in prev_completion.strip().splitlines()[:8])
    return (
        "# Previous attempt was wrong:\n"
        f"{indented}\n"
        f"# Failed test: {short_failure}\n"
        "# Fix the bug and try again.\n\n"
        + prompt
    )
