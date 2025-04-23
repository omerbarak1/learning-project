import random

def generate_synthetic_student():
    def rand_dict(keys):
        return {k: round(random.uniform(0, 1), 2) for k in keys}

    return {
        "mastery": rand_dict(["fractions", "geometry", "percentages"]),
        "psychology": rand_dict(["attention_level", "anxiety_sensitivity", "impulsivity", "patience"]),
        "goals": rand_dict(["grades", "understanding", "speed", "creativity"]),
        "interests": rand_dict(["sports", "puzzles", "history", "music"]),
        "meta": rand_dict(["learning_rate_estimate", "retention", "reward_sensitivity", "avg_response_time", "motivation_level"]),
        "learning_style": rand_dict(["visual", "verbal", "kinesthetic"])
    }
