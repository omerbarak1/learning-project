import random

def flatten_topics(hierarchy, prefix=None):
    """
    Flatten a hierarchical dict of topics into dot-separated keys.
    e.g. {'algebra': {'fractions': None}} -> ['algebra.fractions']
    """
    topics = []
    for k, v in hierarchy.items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict) and v:
            topics.extend(flatten_topics(v, path))
        else:
            topics.append(path)
    return topics

class StudentState:
    """
    Manages a hierarchical student state, including mastery over nested topics.
    """
    def __init__(self, topic_hierarchy=None):
        # 1. Define a generic math curriculum hierarchy if not provided
        self.topic_hierarchy = topic_hierarchy or {
            'arithmetic': {
                'fractions': None,
                'percentages': None,
                'decimals': None
            },
            'algebra': {
                'expressions': None,
                'equations': None,
                'inequalities': None
            },
            'geometry': {
                'triangles': None,
                'circles': None,
                'polygons': None
            },
            'probability_statistics': {
                'probability': None,
                'statistics': None
            },
            'functions': {
                'linear': None,
                'quadratic': None
            },
            'calculus': {
                'limits': None,
                'derivatives': None,
                'integrals': None
            }
        }

        # Initialize state structure
        self.state = {
            'mastery': self._init_mastery(self.topic_hierarchy),
            'psychology': {
                'attention_level': 0.6,
                'anxiety_sensitivity': 0.5,
                'impulsivity': 0.7,
                'patience': 0.4,
            },
            'goals': {
                'grades': 1.0,
                'understanding': 0.8,
                'speed': 0.3,
                'creativity': 0.5,
            },
            'interests': {
                'sports': 0.9,
                'puzzles': 0.8,
                'history': 0.2,
                'music': 0.4,
            },
            'meta': {
                'learning_rate_estimate': 0.7,
                'retention': 0.6,
                'reward_sensitivity': 0.8,
                'avg_response_time': 0.5,
                'motivation_level': 0.75,
            },
            'learning_style': {
                'visual': 0.7,
                'verbal': 0.4,
                'kinesthetic': 0.2,
            }
        }
        # Compute derived topic rankings
        self._update_topic_ranking()

    def _init_mastery(self, hierarchy):
        """Recursively initialize mastery structure mirroring topic_hierarchy."""
        mastery = {}
        for k, v in hierarchy.items():
            if isinstance(v, dict) and v:
                mastery[k] = self._init_mastery(v)
            else:
                mastery[k] = 0.0
        return mastery

    def update_mastery(self, topic_path, success: bool):
        """
        Update mastery for a specific topic path (dot-separated) based on success.
        e.g. topic_path='algebra.equations'
        """
        keys = topic_path.split('.')
        d = self.state['mastery']
        for key in keys[:-1]:
            d = d[key]
        last = keys[-1]
        delta = 0.1 if success else -0.05
        d[last] = max(0.0, min(1.0, d[last] + delta))
        self._update_topic_ranking()

    def _update_topic_ranking(self):
        """Flatten and rank topic mastery for summary and feature generation."""
        flat = {}
        for path in flatten_topics(self.topic_hierarchy):
            d = self.state['mastery']
            for key in path.split('.'):
                d = d[key]
            flat[path] = d
        ranked = sorted(flat.items(), key=lambda kv: kv[1], reverse=True)
        self.state['topics_ranked_list'] = [t for t, _ in ranked]
        self.state['topics_ranked_str'] = ", ".join(self.state['topics_ranked_list'])

    def to_vector(self):
        """Return a flat numeric vector of all features (mastery, psychology, etc.)."""
        vec = []
        # Mastery: flatten hierarchical values
        for path in flatten_topics(self.topic_hierarchy):
            d = self.state['mastery']
            for key in path.split('.'):
                d = d[key]
            vec.append(d)
        # Psychology, goals, interests, meta, learning_style
        for cat in ['psychology', 'goals', 'interests', 'meta', 'learning_style']:
            for v in self.state[cat].values():
                vec.append(v)
        return vec

    def dict(self):
        """Return internal state dict."""
        return self.state
