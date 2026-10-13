"""LLM: load, ask, release + robust JSON extraction (checklist 3.5).

No diffusers/torch-heavy imports here except `torch` for cache release;
keep the LLM phase and the SDXL phase strictly separated by callers
(call `gpu.release_gpu()` between them).
"""
from __future__ import annotations

import ctypes
import json
import re
import time
from typing import Callable

import llama_cpp


# The registered C-log callback must stay alive for the whole process: if a
# CFUNCTYPE object is garbage-collected while C still holds its trampoline,
# the next C-side log call jumps into freed memory (deterministic segfault
# inside _ctypes — the stage-1 crash of 2026-10-12). Module-level reference.
_SILENT_LOG_SINK: "ctypes._CFuncPtr | None" = None


def _silence_ollama_logs() -> None:
    global _SILENT_LOG_SINK
    try:
        @ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)
        def _cb(level, text, user_data):
            pass

        _SILENT_LOG_SINK = _cb  # keep alive for the process lifetime!
        llama_cpp.llama_log_set(_cb, ctypes.c_void_p(0))
    except Exception:
        pass


def create_llm(model_path: str, n_ctx: int = 8192, log=None):
    lg = log or (lambda m: None)
    import os as _os
    try:
        lib_dir = _os.path.join(_os.path.dirname(llama_cpp.__file__), "lib")
        libs = sorted(f for f in _os.listdir(lib_dir)
                      if any(k in f.lower() for k in ("cuda", "cublas", "cudnn", "cufft")))
        lg(f"llama_cpp={_os.path.dirname(llama_cpp.__file__)}")
        lg(f"bundled cuda libs: {libs if libs else 'NONE (CPU-only build!)'}")
    except Exception as e:  # noqa: BLE001 - probe only
        lg(f"env probe failed: {e!r}")
    lg(f"-> installing silent C-log callback")
    _silence_ollama_logs()
    lg(f"-> llama_cpp.Llama(n_gpu_layers=-1, n_ctx={n_ctx}, flash_attn=True) [loading ~16 GB] ...")
    model = llama_cpp.Llama(
        model_path=model_path,
        n_gpu_layers=-1,
        n_ctx=n_ctx,
        flash_attn=True,
        verbose=False,
    )
    lg("Llama loaded OK")
    return model


def ask(llm, system: str, user: str, max_tokens: int = 0,
        temperature: float = 0.2, log=None) -> str:
    """One LLM call. ALWAYS logs: payload size, raw output size, model usage
    (token counts + stop reason), and a raw content slice whenever the
    stripped result is empty. Whatever the failure mode, run.log gets the
    facts instead of a guess.

    max_tokens=0 = unlimited (bounded only by n_ctx): this model thinks
    first and burns ~2k tokens before the actual answer — a fixed 2k cap
    cuts it off mid-JSON and then strip_think eats the open think block.
    """
    lg = log or (lambda m: None)
    payload = (
        f"system\n{system}\n\n"
        f"user\n{user}\n\n"
        "assistant\n"
    )
    lg(f"ask: payload {len(payload)} chars -> {payload[:200]!r} ...")
    lg(f"ask: calling model (max_tokens={max_tokens} temperature={temperature}) ...")
    t0 = time.monotonic()
    out = llm(payload, max_tokens=max_tokens, temperature=temperature,
              repeat_penalty=1.05)
    dt = time.monotonic() - t0
    text = out["choices"][0]["text"]
    usage = out.get("usage") or {}
    lg(f"ask: done in {dt:.1f}s | raw_chars={len(text)} | "
       f"completion_tokens={usage.get('completion_tokens')} "
       f"prompt_tokens={usage.get('prompt_tokens')} | stop_reason={out.get('stop_reason')!r}")
    if text:
        lg(f"ask: raw first 400: {text[:400]!r}")
        lg(f"ask: raw last 400:  {text[-400:]!r}")
    else:
        lg("ask: model returned EMPTY text. Non-text parts of its response:")
        lg(repr({k: v for k, v in out.items() if k != "choices"}))
        lg(repr(out["choices"][0]))
    stripped = strip_think(text)
    lg(f"ask: post-strip chars={len(stripped)} (removed {len(text) - len(stripped)} by strip_think)")
    if not stripped:
        lg("ask: WARNING: no usable output — thinking consumed everything or model produced nothing")
    elif not stripped.rstrip().endswith(("}", "]")):
        lg("ask: WARNING: output does not end with a JSON closer -> truncated at the context limit")
    return stripped


def strip_think(text: str) -> str:
    """Remove Qwen think-block markers if thinking mode leaks into output."""
    open_tag = chr(60) + "think" + chr(62)
    close_tag = chr(60) + "/" + "think" + chr(62)
    text = re.sub(re.escape(open_tag) + r".*?" + re.escape(close_tag),
                  "", text, flags=re.DOTALL)
    text = re.sub(re.escape(open_tag) + r".*$", "", text, flags=re.DOTALL)
    return text.strip()


def extract_json(text: str):
    """Best-effort: fenced block first, then raw, then first balanced bracket scan."""
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        start = text.find(open_ch)
        end = text.rfind(close_ch)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("no parseable JSON in LLM output")


def llm_json(llm, system: str, user: str,
             validate: Callable[[object], object],
             log=None) -> object:
    """Ask → parse → validate. One retry with the concrete error on failure.

    The RAW model output of every attempt is always logged (that is the only
    evidence we get when parsing fails — never let it go unlogged again).
    """
    lg = log or (lambda m: None)

    def _dump(raw: str, tag: str) -> None:
        lg(f"--- LLM raw output {tag} ({len(raw)} chars) ---")
        lg(raw)
        lg(f"--- end {tag} ---")

    raw = ask(llm, system, user, log=lg)
    _dump(raw, "attempt 1")
    try:
        data = extract_json(raw)
        return validate(data)
    except Exception as first_err:  # noqa: BLE001 - one retry, then raise
        lg(f"LLM output invalid ({first_err}): {first_err!r}; re-prompting once")
        retry_user = (
            user
            + "\n\nYour previous answer was invalid:\n"
            + str(first_err)
            + "\nAnswer again with JSON only. No prose."
        )
        raw2 = ask(llm, system, retry_user, log=lg)
        _dump(raw2, "attempt 2")
        data = extract_json(raw2)
        result = validate(data)
        lg("re-prompt succeeded")
        return result
