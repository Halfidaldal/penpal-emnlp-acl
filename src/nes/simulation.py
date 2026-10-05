"""Simulation of the AI-AI condition.

Two instances of the same model co-write a story under the prompt, pacing
metadata and transport used in the Human-AI experiment:

- OpenAI models use server-side conversation threading (Responses API).
- Other models are called statelessly through OpenAI-compatible
  chat/completions endpoints with the rolling history, trimmed to 12k chars.
- A reply that echoes the partner's input has the echo stripped.
- Both agents' outputs are cut by 2-5 trailing words, mirroring the input
  truncation applied to human writers. The truncated text is what is stored
  and what the partner sees.
"""

import os
import random
import time
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

STORY_PREFIX = "This is the story of "

SYSTEM_PROMPT = """You are an author taking part in a collaborative storytelling game activity with another author.
Together, you will create a story by taking turns adding to it.
Your goal is to continue from where your partner has left off.
If there's no story, please begin the story. You have 10 interactions to write the story. Your input may get slightly truncated with a random character amount."""


def system_prompt(turn_number: int) -> str:
    pacing = f"""

[Session Meta — do not reveal]
This is turn {turn_number} of 10. Use this only to pace and conclude appropriately. Do not mention turns, counts, chapters, headings, "#", or session meta in your reply. Write in plain prose that flows naturally from the previous text."""
    return f"{SYSTEM_PROMPT}{pacing}".strip()


def trim_messages_to_char_limit(messages: List[Dict[str, str]], limit: int = 12000) -> List[Dict[str, str]]:
    """Keep the system message plus as many of the most recent messages as fit."""
    if not messages:
        return messages
    system = messages[0] if messages[0].get("role") == "system" else None
    rest = messages[1:] if system else messages[:]

    remaining = limit
    kept: List[Dict[str, str]] = []
    for msg in reversed(rest):
        length = len(msg.get("content", ""))
        if length > remaining and kept:
            continue
        remaining -= length
        kept.insert(0, msg)
        if remaining <= 0:
            break
    return [system] + kept if system else kept


def build_messages(system: Optional[str], history: Optional[List[Dict[str, str]]], user_input: str,
                   char_limit: int = 12000) -> List[Dict[str, str]]:
    history = [dict(m) for m in (history or [])]
    messages: List[Dict[str, str]] = []
    if system and (not history or history[0].get("role") != "system"):
        messages.append({"role": "system", "content": system})
    messages += [m for m in history if (m.get("content") or "").strip()]
    if user_input and user_input.strip():
        messages.append({"role": "user", "content": user_input})
    return trim_messages_to_char_limit(messages, limit=char_limit)


def truncate_words(text: str, n: int) -> str:
    """Drop the last `n` whitespace-separated words."""
    if not isinstance(text, str) or not text.strip():
        return ""
    words = text.strip().split()
    return " ".join(words[:max(0, len(words) - n)]) if n > 0 else text.strip()


def strip_input_echo(response: str, user_input: str) -> str:
    response, user_input = (response or "").strip(), (user_input or "").strip()
    if response and user_input and response.lower().startswith(user_input.lower()):
        response = response[len(user_input):].strip()
    return response


class OpenAIProvider:
    """OpenAI Responses API with server-side conversation threading; `history` is ignored."""

    def __init__(self, api_key: str, model_name: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.conversation_id = None

    def generate(self, system: str, user_input: str, history=None, temperature: float = 1.0,
                 max_tokens: int = 40) -> str:
        if self.conversation_id is None:
            self.conversation_id = self.client.conversations.create(
                metadata={"topic": "ai-ai-simulation"}, items=[]
            ).id
        response = self.client.responses.create(
            model=self.model_name,
            conversation=self.conversation_id,
            instructions=system,
            input=user_input,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        out = response.output_text or ""
        if not out:
            for item in response.output or []:
                content = getattr(item, "content", None)
                if isinstance(content, str):
                    out += content
                elif isinstance(content, list):
                    out += "".join(getattr(c, "text", "") for c in content)
        return out.strip()

    def reset(self):
        self.conversation_id = None


class ChatCompletionsProvider:
    """Stateless OpenAI-compatible chat/completions endpoint (Anthropic, HF router, OpenRouter)."""

    def __init__(self, api_key: str, model_name: str, base_url: str):
        from openai import OpenAI
        if not base_url:
            raise ValueError(f"{model_name}: base_url is required for chat/completions providers")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def generate(self, system: str, user_input: str, history=None, temperature: float = 1.0,
                 max_tokens: int = 40) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=build_messages(system, history, user_input),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        out = response.choices[0].message.content if response.choices else ""
        return out.strip() if out else ""

    def reset(self):
        pass


def get_provider(provider: str, api_key: str, model_name: str, base_url: Optional[str] = None):
    if provider == "openai":
        return OpenAIProvider(api_key, model_name)
    if provider in ("anthropic", "huggingface"):
        return ChatCompletionsProvider(api_key, model_name, base_url)
    raise ValueError(f"Unknown provider: {provider}")


def simulate_single_story(agent_1, agent_2, n_turns: int, story_id: str, model_id: str,
                          temperature: float = 1.0, max_tokens: int = 40,
                          truncate_min: int = 2, truncate_max: int = 5) -> pd.DataFrame:
    """One story; each turn is agent 1 then agent 2. Each agent keeps its own user/assistant history."""
    agent_1.reset()
    agent_2.reset()
    history_1: List[Dict[str, str]] = []
    history_2: List[Dict[str, str]] = []
    partner_text = ""
    rows = []

    for turn in range(1, n_turns + 1):
        system = system_prompt(turn)

        input_1 = STORY_PREFIX if turn == 1 else partner_text
        text_1 = agent_1.generate(system, input_1, history_1 or None, temperature, max_tokens)
        text_1 = truncate_words(strip_input_echo(text_1, input_1), random.randint(truncate_min, truncate_max))
        history_1 += [{"role": "user", "content": input_1}, {"role": "assistant", "content": text_1}]

        text_2 = agent_2.generate(system, text_1, history_2 or None, temperature, max_tokens)
        text_2 = truncate_words(strip_input_echo(text_2, text_1), random.randint(truncate_min, truncate_max))
        history_2 += [{"role": "user", "content": text_1}, {"role": "assistant", "content": text_2}]
        partner_text = text_2

        rows.append({
            "turn": turn,
            # Stored with the prefix, like the first human turn in Human-AI.
            "agent_1": f"{STORY_PREFIX}\n{text_1}" if turn == 1 else text_1,
            "agent_2": text_2,
            "story_id": story_id,
            "model_id": model_id,
            "timestamp": pd.Timestamp.now(),
        })
    return pd.DataFrame(rows)


def retry_with_backoff(func, max_retries: int = 5, base_delay: float = 2.0):
    """Retry `func` on rate-limit errors with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            message = str(e).lower()
            if not any(s in message for s in ("429", "rate limit", "rate_limit", "too many requests")):
                raise
            if attempt == max_retries - 1:
                raise
            delay = base_delay * 2 ** attempt + random.uniform(0, 1)
            print(f"Rate limited, retrying in {delay:.1f}s ({attempt + 1}/{max_retries})")
            time.sleep(delay)


def simulate_ai_ai_dataset(model_configs: List[Dict], n_stories_per_model: int = 10, n_turns_per_story: int = 10,
                           temperature: float = 1.0, max_tokens: int = 40,
                           delay_between_stories: float = 2.0) -> pd.DataFrame:
    """Generate stories round-robin across models (one story per model at a time)."""
    models = []
    for cfg in model_configs:
        api_key = os.environ.get(cfg["env_key"])
        if not api_key:
            print(f"Skipping {cfg['id']}: {cfg['env_key']} not set")
            continue
        make = lambda: get_provider(cfg["provider"], api_key, cfg["model_name"], cfg.get("base_url"))  # noqa: E731
        models.append({"config": cfg, "agents": (make(), make()), "done": 0})
    if not models:
        return pd.DataFrame()

    stories = []
    with tqdm(total=n_stories_per_model * len(models), desc="Stories") as pbar:
        while any(m["done"] < n_stories_per_model for m in models):
            for m in models:
                if m["done"] >= n_stories_per_model:
                    continue
                model_id = m["config"]["id"]
                story_id = f"{model_id}_story_{m['done'] + 1}"
                try:
                    stories.append(retry_with_backoff(lambda: simulate_single_story(
                        *m["agents"], n_turns_per_story, story_id, model_id, temperature, max_tokens
                    )))
                except Exception as e:
                    print(f"Failed {story_id}: {e}")
                m["done"] += 1
                pbar.update(1)
                if any(x["done"] < n_stories_per_model for x in models):
                    time.sleep(delay_between_stories)

    return pd.concat(stories, ignore_index=True) if stories else pd.DataFrame()
