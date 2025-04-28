# src/scripts/synthesize_data.py

import json
import random
from src.student_state import StudentState               # use your StudentState class
from src.utils.flatten import flatten_dict

OUTPUT_PATH = "src/data/synthetic_state_prompt_ext.jsonl"
N_EXAMPLES = 5000

# Define various prompt templates incorporating state

def template_basic(topic, difficulty, state, mastery_val):
    return f"Please solve a level {difficulty} {topic} problem."


def template_mastery_focus(topic, difficulty, state, mastery_val):
    return (f"Student's mastery in {topic} is {mastery_val:.2f}. "
            f"Create a level {difficulty} problem to further develop this skill.")


def template_interest_blend(topic, difficulty, state, mastery_val):
    interests = state.get("interests", {})
    if interests:
        choice = random.choice(list(interests.keys()))
        return (f"Since the student enjoys {choice}, craft a level {difficulty} {topic} word problem "
                f"that incorporates {choice}.")
    return template_basic(topic, difficulty, state, mastery_val)


def template_attention_short(topic, difficulty, state, mastery_val):
    att = state.get("psychology", {}).get("attention_level", 1.0)
    if att < 0.5:
        return (f"Attention seems low ({att:.2f}), so give a very short level {difficulty} "
                f"{topic} question.")
    else:
        return template_basic(topic, difficulty, state, mastery_val)


def template_goal_driven(topic, difficulty, state, mastery_val):
    speed_goal = state.get("goals", {}).get("speed", 0.0)
    if speed_goal > 0.5:
        return (f"Student aims for speed ({speed_goal:.2f}). "
                f"Generate a timed level {difficulty} {topic} challenge.")
    return template_basic(topic, difficulty, state, mastery_val)

# List of available templates
TEMPLATES = [
    template_basic,
    template_mastery_focus,
    template_interest_blend,
    template_attention_short,
    template_goal_driven,
]


def generate_example():
    # 1) Instantiate StudentState and get state dict
    student = StudentState()
    state = student.state

    # 2) Flatten mastery and avoid zero values
    mastery_flat = flatten_dict(state['mastery'])
    mastery_flat = {k: (v if v > 0 else 0.01) for k, v in mastery_flat.items()}

    # 3) Choose a random topic
    chosen_topic = random.choice(list(mastery_flat.keys()))
    mastery_val = mastery_flat[chosen_topic]

    # 4) Sample difficulty and time
    difficulty = random.randint(1, 5)          # integer difficulty 1–5
    time_spent = random.uniform(10, 120)       # time in seconds

    # 5) Determine correctness probability based on mastery & difficulty
    p = mastery_val * ((6.0 - difficulty))
    p = max(0.0, min(1.0, p))
    correctness = random.uniform(0,1-p)

    # 6) Compute reward
    reward = (1.0 / mastery_val) * (1.0 / time_spent) * correctness * difficulty

    # 7) Build prompt using a random template
    template = random.choice(TEMPLATES)
    prompt = template(chosen_topic, difficulty, state, mastery_val)

    return {
        "state":       state,
        "prompt":      prompt,
        
        # "difficulty":  difficulty,
        # "time":        time_spent,
        # "correctness": correctness,
        
        "reward":      reward,
    }


def main():
    random.seed(42)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for _ in range(N_EXAMPLES):
            ex = generate_example()
            out.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Wrote {N_EXAMPLES} examples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
