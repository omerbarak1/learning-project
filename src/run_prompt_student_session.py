import random
import time
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
def get_topic_from_prompt(prompt: str) -> str:
    for word in prompt.split():
        if word.lower() in ['fractions', 'geometry', 'percentages']:
            return word.lower()
    return 'general'

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
            if cat == 'topics_ranked_str':
                parts.append(f"topics_ranked: {state['topics_ranked_str']}")
            elif isinstance(feats, dict):
                entries = ", ".join(f"{k}={v}" for k,v in feats.items())
                parts.append(f"{cat}: {entries}")
        return " | ".join(parts)

    def generate(self, state: dict, action_idx: int, context: str=None) -> str:
        state_text = self.serialize_state(state)
        action_name = PromptAction(action_idx).name
        input_parts = [f"STATE: {state_text}"]
        if context and action_name == 'HINT':
            input_parts.append(f"CONTEXT: {context}")
        input_parts.append(f"ACTION: {action_name}")
        suffix = 'HINT:' if action_name == 'HINT' else 'PROMPT:'
        input_parts.append(suffix)
        input_text = " | ".join(input_parts)
        inputs = self.tokenizer(input_text, return_tensors="pt")
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            top_p=self.top_p,
            pad_token_id=self.tokenizer.eos_token_id
        )
        gen_tokens = out[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

if __name__ == '__main__':
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Initialize student and numeric state vector
    student = StudentState()
    flat = flatten_dict(student.state)
    state_vec = [v for v in flat.values() if isinstance(v,(int,float))]
    state_dim = len(state_vec)

    # Initialize action agent
    n_actions = len(PromptAction)
    agent = DeepPromptAgent(
        state_dim=state_dim,
        n_actions=n_actions,
        templates_per_action=[1]*n_actions,
        gamma=0.99, lr=1e-3
    )

    # Initialize prompt generator
    prompt_gen = PromptGenerator(model_dir="ft-prompt-agent")

    # Variables for conditional hint and deferred reward
    hint_delay = 30.0  # seconds
    last_question_time = None
    last_question_prompt = None
    pending_transition = None  # (prev_state_vec, prev_action_idx)

    topic_history = []
    episodes = 50

    for ep in range(episodes):
        print(f"\n📘 Episode {ep+1}")
        now = time.time()

        # Determine if hint allowed
        can_hint = last_question_time is not None and (now - last_question_time) >= hint_delay

        # Select action with exclusion of HINT if not allowed
        state_tensor = torch.FloatTensor(state_vec).unsqueeze(0)
        q_actions, _ = agent.forward(state_tensor)
        q_vals = q_actions.detach().cpu().numpy()[0]
        allowed = list(range(n_actions))
        if not can_hint:
            allowed.remove(PromptAction.HINT.value)
        # ε-greedy among allowed
        if random.random() < agent.eps:
            action_idx = random.choice(allowed)
        else:
            masked = [q_vals[i] if i in allowed else -float('inf') for i in range(n_actions)]
            action_idx = int(torch.tensor(masked).argmax().item())
        action = PromptAction(action_idx)

        # Generate prompt or hint
        context = last_question_prompt if action == PromptAction.HINT else None
        prompt = prompt_gen.generate(student.state, action_idx, context)
        print("DEBUG prompt =", prompt)

        # Record question time and setup deferred hint reward
        if action == PromptAction.QUESTION:
            last_question_time = now
            last_question_prompt = prompt
            # If pending hint/explanation, settle its reward using this question's outcome
            if pending_transition:
                prev_state, prev_action = pending_transition
                success = None  # will compute after simulation
                # simulate now for this question
                success, difficulty = random.random() < 0.5, random.uniform(0.4,1.0)
                topic = get_topic_from_prompt(prompt)
                novelty = compute_topic_novelty(topic, topic_history + [topic])
                hint_reward = compute_reward(success, difficulty, novelty, prev_action)
                # push and update for hint/explanation
                agent.push_transition(prev_state, prev_action, 0, hint_reward, state_vec)
                agent.update()
                pending_transition = None

        # Send to GPT for instructional content
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role":"system","content":"You are a helpful math teacher."},
                {"role":"user","content":prompt}
            ],
            temperature=0.7
        )
        output = response.choices[0].message.content
        print(f"\n🤖 GPT Output ({action.name}):\n{output.strip()}")

        # For non-question actions, defer reward
        if action in [PromptAction.HINT, PromptAction.EXPLANATION]:
            pending_transition = (state_vec, action_idx)
        else:
            # Simulate student and compute reward for question or other actions
            success = random.random() < 0.5
            difficulty = random.uniform(0.4,1.0)
            topic = get_topic_from_prompt(prompt)
            topic_history.append(topic)
            novelty = compute_topic_novelty(topic, topic_history)
            reward = compute_reward(success, difficulty, novelty, action_idx)
            # push and update for this action
            agent.push_transition(state_vec, action_idx, 0, reward, state_vec)
            agent.update()

        # Update student mastery and state_vec
        if action == PromptAction.QUESTION:
            student.update_mastery(get_topic_from_prompt(prompt), success)
        flat = flatten_dict(student.state)
        state_vec = [v for v in flat.values() if isinstance(v,(int,float))]

    print("Training complete.")
