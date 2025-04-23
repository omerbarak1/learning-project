import random
import torch
from dotenv import load_dotenv
from openai import OpenAI
import os

from src.agents.deep_prompt_agent import DeepPromptAgent
from src.agents.prompt_action import PromptAction
from src.utils.flatten import flatten_dict
from src.utils.reward import compute_reward, compute_topic_novelty
from src.student_state import StudentState
from transformers import AutoTokenizer, AutoModelForCausalLM

# Extract topic from prompt text
# Ensure get_topic_from_prompt is defined

def get_topic_from_prompt(prompt: str) -> str:
    for word in prompt.split():
        if word.lower() in ['fractions', 'geometry', 'percentages']:
            return word.lower()
    return 'general'

# PromptGenerator: generates prompt text using a fine-tuned language model
class PromptGenerator:
    def __init__(self, model_dir="ft-prompt-agent", max_new_tokens=100, top_p=0.9):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model     = AutoModelForCausalLM.from_pretrained(model_dir)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p

    def serialize_state(self, state: dict) -> str:
        parts = []
        for cat, feats in state.items():
            entries = ", ".join(f"{k}={v}" for k, v in feats.items())
            parts.append(f"{cat}: {entries}")
        return " | ".join(parts)

    def generate(self, state: dict, action_idx: int) -> str:
        # Build input text containing state and action
        state_text = self.serialize_state(state)
        action_name = PromptAction(action_idx).name
        input_text = f"STATE: {state_text} | ACTION: {action_name}\nPROMPT:"
        inputs = self.tokenizer(input_text, return_tensors="pt")
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            top_p=self.top_p,
            pad_token_id=self.tokenizer.eos_token_id
        )
        # Decode only the newly generated portion
        gen_tokens = out[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

# Main RL loop: action-agent selects intervention, PromptGenerator crafts text
if __name__ == '__main__':
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Initialize student and numeric state vector
    student = StudentState()
    flat = flatten_dict(student.state)
    state_vec = [v for v in flat.values() if isinstance(v, (int, float))]
    state_dim = len(state_vec)

    # Initialize DeepPromptAgent (action selection only)
    n_actions = len(PromptAction)
    agent = DeepPromptAgent(
        state_dim=state_dim,
        n_actions=n_actions,
        templates_per_action=[1]*n_actions,  # dummy template head sizes
        gamma=0.99, lr=1e-3
    )

    # Initialize language-model-based prompt generator
    prompt_gen = PromptGenerator(model_dir="ft-prompt-agent")

    topic_history = []
    episodes = 50

    for ep in range(episodes):
        print(f"\n📘 Episode {ep+1}")

        # 1) select high-level action
        action_idx, _ = agent.select(state_vec)
        action = PromptAction(action_idx)

        # 2) generate prompt text via LM
        prompt = prompt_gen.generate(student.state, action_idx)
        print("DEBUG prompt =", prompt)

        # 3) call external GPT for content (optional)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful math teacher."},
                {"role": "user",   "content": prompt}
            ],
            temperature=0.7
        )
        output = response.choices[0].message.content
        print(f"\n🤖 GPT Output ({action.name}):\n{output.strip()}")

        # 4) simulate student and compute reward
        success = random.random() < 0.5
        difficulty = random.uniform(0.4, 1.0)
        topic = get_topic_from_prompt(prompt)
        topic_history.append(topic)
        novelty = compute_topic_novelty(topic, topic_history)
        reward = compute_reward(success, difficulty, novelty, action_idx)
        print(f"✅ Success: {success} | 🎯 Difficulty: {difficulty:.2f} | 🔄 Novelty: {novelty:.2f} | 💰 Reward: {reward:.2f}")

        # 5) update student mastery (optional)
        student.update_mastery(topic, success)
        flat = flatten_dict(student.state)
        state_vec = [v for v in flat.values() if isinstance(v, (int, float))]

        # 6) store transition and update action-agent
        agent.push_transition(state_vec, action_idx, 0, reward, state_vec)
        agent.update()

    print("Training complete.")
