from enum import Enum

class PromptAction(Enum):
    QUESTION = 0
    EXPLANATION = 1
    HINT = 2
    REINFORCEMENT = 3
    TEACH_NEW = 4

# Diverse prompt templates to bootstrap the LM
PROMPT_TEMPLATES = {
    PromptAction.QUESTION: [
        "Generate a clear and concise question on {topic}, given mastery: {mastery}, interests: {interests}, goals: {goals}, and psychology: {psychology}.",
        "Create a challenging {topic} problem for a student who has mastery levels: {mastery} and aims: {goals}.",
        "Design a {topic} question that aligns with the student's interests ({interests}) and learning style ({learning_style}).",
        "Formulate a real-world scenario requiring {topic}, appropriate for mastery at: {mastery}."
    ],
    PromptAction.EXPLANATION: [
        "Provide a step-by-step explanation of the {topic} concept, tailored to mastery: {mastery} and psychological traits: {psychology}.",
        "Explain the underlying principles of {topic} using an example that matches interests: {interests}.",
        "Give a detailed walkthrough of how to solve a typical {topic} problem, considering the student's goals: {goals} and patience: {psychology_patience}.",
        "Clarify the key ideas in {topic} with a focus on retention: {meta_retention} and learning style: {learning_style}."
    ],
    PromptAction.HINT: [
        "Offer a targeted hint for the recent {topic} question: {context}, focusing on the next logical step.",
        "Provide a brief tip to approach {topic} efficiently, given the student's impulsivity: {psychology_impulsivity}.",
        "Suggest a strategy to simplify the {topic} problem, leveraging the student's strengths in {topics_ranked_str}.",
        "Give a mnemonic or visualization hint for {topic} that aligns with the student's visual learning preference: {learning_style_visual}."
    ],
    PromptAction.REINFORCEMENT: [
        "Offer a motivational message praising the student's success in {topic}, considering motivation level: {meta_motivation_level} and reward sensitivity: {meta_reward_sensitivity}.",
        "Give positive reinforcement for mastering {topic}, encouraging continued progress toward goals: {goals}.",
        "Celebrate the student's achievement in {topic} and remind them of how their interests ({interests}) connect to the skill.",
        "Provide an encouraging note that reduces anxiety sensitivity: {psychology_anxiety_sensitivity} while acknowledging success in {topic}."
    ],
    PromptAction.TEACH_NEW: [
        "Introduce a new concept in {topic} at the right difficulty for mastery: {mastery}, using an engaging example from {interests}.",
        "Teach the basics of an advanced aspect of {topic}, tailored to the student's speed: {goals_speed} and creativity: {goals_creativity}.",
        "Present an interactive activity to explore a new {topic} topic, aligned with the student's learning style: {learning_style}.",
        "Suggest a real-life application of {topic} that connects to interests: {interests} and enhances understanding: {goals_understanding}."
    ]
}
