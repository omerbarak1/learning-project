import os
import sys
import json
import random

# Ensure we can import the project as a package, regardless of where this script lives
script_dir = os.path.dirname(__file__)
# Go up two levels: from scripts/ or src/scripts/ back to project root
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.student_state import StudentState
from src.utils.flatten import flatten_dict
from src.agents.prompt_action import PromptAction, PROMPT_TEMPLATES


def synthesize_data(n_samples=5000, out_path="data/synthetic_state_prompt.jsonl"):
    """
    Generate synthetic State→Prompt examples using existing templates.

    Args:
        n_samples (int): Number of examples to generate.
        out_path (str): Path to output JSONL file.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Initialize a fresh student state
    student = StudentState()

    with open(out_path, 'w', encoding='utf-8') as f:
        for i in range(n_samples):
            # 1) Randomly update mastery of one topic
            topic = random.choice(student.state['topics_ranked_list'])
            success = random.random() < 0.5
            student.update_mastery(topic, success)

            # 2) Flatten state and build summaries for placeholders
            flat = flatten_dict(student.state)
            summaries = {
                'mastery': student.state['topics_ranked_str'],
                'interests': ', '.join(f"{k.split('_',1)[1]}: {v}" for k,v in flat.items() if k.startswith('interests_')),
                'goals': ', '.join(f"{k.split('_',1)[1]}: {v}" for k,v in flat.items() if k.startswith('goals_')),
                'psychology': ', '.join(f"{k.split('_',1)[1]}: {v}" for k,v in flat.items() if k.startswith('psychology_')),
                'learning_style': ', '.join(f"{k.split('_',1)[1]}: {v}" for k,v in flat.items() if k.startswith('learning_style_')),
                'meta': ', '.join(f"{k.split('_',1)[1]}: {v}" for k,v in flat.items() if k.startswith('meta_')),
                'topics_ranked_str': student.state['topics_ranked_str']
            }
            # 3) Short features without prefixes
            short_feats = {k.split('_',1)[1]: v for k,v in flat.items() if '_' in k}

            # 4) Combine all for formatting
            fmt = {}
            fmt.update(flat)
            fmt.update(short_feats)
            fmt.update(summaries)
            # include topic placeholder for templates
            fmt['topic'] = topic
            # Provide placeholder for context (for HINT templates)
            fmt['context'] = ''

            # 5) Choose action and random template
            action = random.choice(list(PromptAction))
            template = random.choice(PROMPT_TEMPLATES[action])

            # 6) Format prompt text
            prompt = template.format(**fmt)

            # 7) Write out JSONL record
            record = {'state': student.state, 'prompt': prompt}
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"Synthetic data written to {out_path}")


if __name__ == '__main__':
    synthesize_data()
