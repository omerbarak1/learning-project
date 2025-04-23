def compute_reward(success, difficulty, topic_novelty, action_type=None):
    '''
    success: bool (אם התלמיד הצליח)
    difficulty: float [0, 1]
    topic_novelty: float [0, 1] - האם הנושא חדש עבור התלמיד
    action_type: PromptAction (אופציונלי - נוכל בעתיד לתעדף הסברים, חיזוקים וכו')

    return: float - תגמול סופי
    '''
    reward = 0.0

    # בסיס: הצלחה
    reward += 1.0 if success else -0.5

    # קושי תורם אם הצלחה
    if success:
        reward += 0.5 * difficulty

    # גיוון/חדשנות תורם תמיד
    reward += 0.3 * topic_novelty

    # אפשר להוסיף שקלול לפי סוג פעולה בהמשך
    return reward


def compute_topic_novelty(current_topic, topic_history):
    '''
    current_topic: str - הנושא של השאלה הנוכחית
    topic_history: list of str - היסטוריית נושאים ששאלנו עליהם

    return: float ∈ [0, 1] - מדד חדשנות (1 = נושא חדש לחלוטין)
    '''
    if not topic_history:
        return 1.0  # נושא ראשון - חדש לחלוטין

    freq = topic_history.count(current_topic)
    novelty = 1.0 / (1 + freq)  # ככל שהנושא חוזר יותר → הערך קטן יותר

    return novelty
