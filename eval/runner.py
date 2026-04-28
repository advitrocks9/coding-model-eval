"""Generate completions for HumanEval prompts using a local HF model.

Two modes:
    completion: feed the HumanEval prompt verbatim, generate continuation.
    retry:      prepend a short comment about the previous failure, then
                feed the HumanEval prompt again and generate.

Mellum-4b-sft-python is a FIM code-completion model, not a chat model.
Treating the HumanEval prompt as a code prefix and continuing it is the
honest way to evaluate it.
"""
from __future__ import annotations

import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Stop the completion as soon as the model starts writing a new top-level
# def/class or a triple-quoted block that isn't part of the function body.
# We keep this conservative: HumanEval solutions are usually one function.
_TRUNC_PATTERNS = (
    "\ndef ",
    "\nclass ",
    "\nif __name__",
    "\nprint(",
    "\nassert ",
    "\n#",
)


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
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=dtype
        ).to(device).eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def complete(self, prompt: str, temperature: float = 0.0, seed: int | None = None) -> str:
        """Generate continuation. temperature=0 is greedy (default).

        For multi-turn retries we use temperature > 0 with a fixed seed: a
        small prompt change with greedy decoding often produces the same
        argmax sequence (we observed this with `# previous attempt wrong`
        prepended to a HumanEval prompt).
        """
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
        """Fill-in-the-middle generation using Mellum's native FIM tokens.

        Mellum-4b-base is trained to predict text between <fim_prefix> and
        <fim_suffix> wrapped around the surrounding context. The format
        from the model card:

            <filename>name.py
            <fim_suffix>{after}<fim_prefix>{before}<fim_middle>

        The model continues from <fim_middle>. We strip the special tokens
        out of the decoded text. Greedy only.
        """
        wrapped = f"<filename>{filename}\n<fim_suffix>{suffix}<fim_prefix>{prefix}<fim_middle>"
        ids = self.tokenizer(wrapped, return_tensors="pt", return_token_type_ids=False).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        gen = out[0, ids["input_ids"].shape[1]:]
        # don't skip special tokens during decode so we can detect EOM markers
        text = self.tokenizer.decode(gen, skip_special_tokens=False)
        # the model often emits the next FIM block boundary or filename
        # marker as a stop signal; cut there
        for marker in ("<filename>", "<fim_suffix>", "<fim_prefix>", "<|endoftext|>"):
            i = text.find(marker)
            if i >= 0:
                text = text[:i]
                break
        return _truncate(text)


def _truncate(text: str) -> str:
    """Stop at the first thing that looks like a new top-level statement.

    Models often keep writing test code or a second function after the one
    we want. Truncate so the sandbox doesn't accidentally execute that.
    """
    cuts = [text.find(p) for p in _TRUNC_PATTERNS]
    cuts = [c for c in cuts if c >= 0]
    if cuts:
        text = text[: min(cuts)]
    # also strip trailing whitespace; keep the newline so concat is clean
    return text.rstrip() + "\n"


def build_retry_prompt(
    prompt: str, prev_completion: str, failure: str
) -> str:
    """Prepend a code comment about the failed attempt.

    This is what an IDE's chat-with-current-file would naturally produce
    when asked to retry: a short hint comment right above the function the
    user is editing. We do not feed back the full traceback (per rescue
    feedback: noisy, leads to regressions on small models).
    """
    short_failure = re.sub(r"\s+", " ", failure).strip()[:160]
    indented = "\n".join("# " + line for line in prev_completion.strip().splitlines()[:8])
    hint = (
        "# Previous attempt was wrong:\n"
        f"{indented}\n"
        f"# Failed test: {short_failure}\n"
        "# Fix the bug and try again.\n\n"
    )
    return hint + prompt
