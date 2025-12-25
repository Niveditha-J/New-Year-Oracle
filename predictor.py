"""Simple heuristic-based emotion and prediction engine for New Year Oracle.
This uses MediaPipe FaceMesh landmarks to compute crude mouth/eye/head metrics,
then maps them to playful, weighted predictions. No training, no sensitive inference.
"""
import random
from typing import List, Dict

import numpy as np
try:
    import mediapipe as mp
except Exception:
    mp = None


def analyze_image(pil_image, user_text: str = "") -> Dict:
    """Analyze a PIL image, return mood, confidence, and a list of predictions.

    Returns a dict like:
    {
      'mood': 'joyful',
      'confidence': 0.84,
      'predictions': [ {'title': 'Career upgrade detected 💼', 'detail': '...'}, ... ]
    }
    """
    if mp is None:
        return _fallback_response(user_text)

    img = np.array(pil_image)
    h, w = img.shape[:2]

    mp_face = mp.solutions.face_mesh
    with mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as fm:
        results = fm.process(img[:, :, ::-1])
        if not results.multi_face_landmarks:
            return _no_face_response(user_text)

        lm = results.multi_face_landmarks[0].landmark
        pts = np.array([[p.x * w, p.y * h] for p in lm])

    def idxs_from_connections(conns):
        s = set()
        for a, b in conns:
            s.add(a); s.add(b)
        return sorted(s)

    left_eye_idx = idxs_from_connections(mp_face.FACEMESH_LEFT_EYE)
    right_eye_idx = idxs_from_connections(mp_face.FACEMESH_RIGHT_EYE)
    lips_idx = idxs_from_connections(mp_face.FACEMESH_LIPS)

    # face bounding box
    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)
    face_w = max_xy[0] - min_xy[0]
    face_h = max_xy[1] - min_xy[1]

    mouth = pts[lips_idx]
    mouth_h = mouth[:, 1].max() - mouth[:, 1].min()
    mouth_w = mouth[:, 0].max() - mouth[:, 0].min()
    mouth_open = mouth_h / (face_h + 1e-6)
    smile_score = mouth_w / (face_w + 1e-6)

    le = pts[left_eye_idx]
    re = pts[right_eye_idx]
    eye_open_left = (le[:, 1].max() - le[:, 1].min()) / (face_h + 1e-6)
    eye_open_right = (re[:, 1].max() - re[:, 1].min()) / (face_h + 1e-6)
    eye_open = (eye_open_left + eye_open_right) / 2.0

    left_centroid = le.mean(axis=0)
    right_centroid = re.mean(axis=0)
    head_tilt = (left_centroid[1] - right_centroid[1]) / (left_centroid[0] - right_centroid[0] + 1e-6)

    # crude confidence: larger, clearer faces yield higher confidence
    face_proportion = min(1.0, face_w / (w + 1e-6))
    confidence = 0.4 + 0.6 * face_proportion
    confidence = float(max(0.05, min(0.99, confidence)))

    # decide mood using simple rules
    mood = 'neutral'
    emoji = ''
    note = ''
    if smile_score > 0.38 and eye_open > 0.03:
        mood = 'joyful'; emoji = '😊'; note = 'Warm, open expression'
    elif mouth_open > 0.07:
        mood = 'surprised'; emoji = '😲'; note = 'Mouth open — surprised or excited'
    elif eye_open < 0.018:
        mood = 'tired'; emoji = '😴'; note = 'Low eye openness — relaxed or tired'
    elif abs(head_tilt) > 0.09:
        mood = 'playful'; emoji = '😉'; note = 'Head tilt detected — playful energy'
    elif smile_score > 0.32:
        mood = 'confident'; emoji = '😎'; note = 'Confident smile'
    else:
        mood = 'calm'; emoji = '😌'; note = 'Calm and steady'

    predictions = _generate_predictions(mood, confidence, user_text)

    return {
        'mood': mood,
        'emoji': emoji,
        'confidence': confidence,
        'note': note,
        'predictions': predictions,
    }


def _weighted_sample(items: List, weights: List[float], k: int = 1):
    chosen = []
    pool = items.copy()
    w = weights.copy()
    for _ in range(min(k, len(pool))):
        pick = random.choices(pool, weights=w, k=1)[0]
        i = pool.index(pick)
        chosen.append(pick)
        # remove picked item
        pool.pop(i); w.pop(i)
    return chosen


def _generate_predictions(mood: str, confidence: float, user_text: str) -> List[Dict]:
    # weighted outputs per mood
    mapping = {
        'joyful': [
            ("Career upgrade detected 💼", 1),
            ("New relationship loading… ❤️", 2),
            ("Money flow stable and growing 💰", 2),
            ("Major personal glow-up incoming ✨", 1),
        ],
        'confident': [
            ("New opportunities on the horizon 💼", 2),
            ("A bold project pays off 💡", 2),
            ("Recognition is coming 🌟", 1),
        ],
        'playful': [
            ("Unexpected fun encounter 🎉", 3),
            ("Creative streak ignites ✨", 2),
            ("A spontaneous trip possible 🧳", 1),
        ],
        'tired': [
            ("Slow start, strong finish ⏳", 3),
            ("Rest first, then bloom 🌱", 2),
            ("Small wins stack up 💪", 1),
        ],
        'surprised': [
            ("A surprise opportunity appears 🎁", 3),
            ("An unexpected message matters ✉️", 2),
            ("Sudden insight fuels change ⚡", 1),
        ],
        'calm': [
            ("Steady progress at your pace 🧭", 3),
            ("Financial balance improves slowly 💰", 2),
            ("Inner growth, outer changes follow 🌿", 1),
        ],
        'neutral': [
            ("Subtle shifts build momentum 🌀", 3),
            ("Network interactions bear fruit 🤝", 2),
            ("Small risks, notable gains 📈", 1),
        ],
    }

    choices = mapping.get(mood, mapping['neutral'])
    titles = [c[0] for c in choices]
    base_weights = [float(c[1]) for c in choices]

    # add some randomness/jitter to weights so similar inputs can yield different picks
    # stronger jitter so repeated photos produce more varied outcomes
    jittered = []
    for w in base_weights:
        factor = 1.0 + random.uniform(-0.8, 0.8)
        jittered.append(max(0.01, w * factor))

    # sample up to 3 non-repeating outcomes using jittered weights
    picked = _weighted_sample(titles, jittered, k=3)
    # ensure variety: shuffle picked so the order isn't deterministic
    random.shuffle(picked)

    out = []
    for t in picked:
        detail = _make_detail(t, mood, confidence, user_text)
        out.append({'title': t, 'detail': detail})
    return out


def _make_detail(title: str, mood: str, confidence: float, user_text: str) -> str:
    templates = [
        "Based on patterns seen in people with similar expressions — be open to small nudges.",
        "Small steps this quarter lead to a big moment later in the year.",
        "Focus your energy; subtle shifts will compound into visible gains.",
    ]
    note = random.choice(templates)
    if user_text and len(user_text.strip()) > 6:
        snippet = user_text.strip().split('.')[:1][0]
        return f"{note} (You mentioned: ‘{snippet}’)"
    return note


def _no_face_response(user_text: str = "") -> Dict:
    # If no face detected, give playful fallback
    predictions = [
        {'title': 'Camera skipped — try again 📷', 'detail': 'No face detected; please retry.'},
        {'title': 'Energy remains mysterious ✨', 'detail': 'Try a different angle or light.'},
    ]
    return {'mood': 'no_face', 'emoji': '🤖', 'confidence': 0.12, 'note': 'No face detected', 'predictions': predictions}


def _fallback_response(user_text: str = "") -> Dict:
    # mediapipe not available
    choices = [
        {'title': 'Career upgrade detected 💼', 'detail': 'Small signals suggest professional momentum.'},
        {'title': 'New relationship loading… ❤️', 'detail': 'Social opportunities appear likely.'},
        {'title': 'Major personal glow-up incoming ✨', 'detail': 'Work on self-care; results follow.'},
    ]
    return {'mood': 'fallback', 'emoji': '✨', 'confidence': 0.2, 'note': 'Mediapipe unavailable — using fallback', 'predictions': choices}


def quick_suggestion(user_text: str = "") -> Dict:
    """Return a fast, lightweight suggestion for immediate overlay while full analysis runs.

    This intentionally uses a small random pick and should be replaced by the full analysis.
    """
    # Backwards-compatible single suggestion using the full quick list
    items = quick_suggestions(1, user_text)
    return items[0]


def quick_suggestions(n: int = 25, user_text: str = "") -> List[Dict]:
    """Return up to `n` quick recommendations (playful, lightweight).

    These are intended as immediate overlay options while the full
    MediaPipe analysis runs. Returns a list of dicts with `title`, `emoji`,
    and `detail`.
    """
    pool = [
        ("New relationship loading… ❤️", "A serendipitous connection is likely."),
        ("Career upgrade detected 💼", "Signals indicate professional momentum."),
        ("Money flow unstable but improving 💰", "Cashflow oscillates but trends upward."),
        ("Major personal glow-up incoming ✨", "Small changes compound into big looks."),
        ("Slow start, strong finish ⏳", "Take it easy early — energy returns later."),
        ("Creative streak begins 🎨", "Ideas flow; capture them quickly."),
        ("Unexpected travel on the horizon 🧳", "A short trip or day escape may appear."),
        ("Friendship rekindled 🤝", "An old contact may reach out with warmth."),
        ("Side project payoff 💡", "A hobby or side gig shows promise."),
        ("Micro-moment of clarity ⚡", "A tiny insight unlocks progress."),
        ("Networking luck 🍀", "Say yes to invites this season."),
        ("Health reset opportunity 🥗", "Small habits improve wellbeing."),
        ("A bold ask pays off 📣", "Courageous requests have favorable odds."),
        ("Financial tidy-up suggested 🧾", "Organize finances for peace of mind."),
        ("Learning curve — rewarding 📚", "Pick one skill to level up this year."),
        ("Household surprise 🏠", "Home improvements bring unexpected joy."),
        ("Romantic spark possible 💘", "A small gesture could ignite chemistry."),
        ("Creative collaboration awaits 🎭", "Team up for a fun, visible result."),
        ("Quiet reflection benefits 🧘", "A short pause will clarify priorities."),
        ("Public recognition likely 🌟", "Work you share could get noticed."),
        ("Bold pivot opportunity 🔀", "A change of course may open doors."),
        ("Generosity returns ❤️", "Giving time or help circles back."),
        ("Digital detox helps 🔌", "Less screen time gives clearer thinking."),
        ("Surprise message matters ✉️", "Check that unexpected note soon."),
        ("Minor purchase, major joy 🎁", "A small buy brings disproportionate pleasure."),
    ]

    # choose up to n random, non-repeating suggestions
    k = min(n, len(pool))
    sampled = random.sample(pool, k)
    out = []
    for title, detail in sampled:
        out.append({'title': title, 'emoji': '', 'detail': detail})

    # personalize slightly if the user supplied text
    if user_text and len(user_text.strip()) > 6:
        snippet = user_text.strip().split('.')[:1][0]
        for i in range(len(out)):
            out[i]['detail'] = f"{out[i]['detail']} (You said: ‘{snippet}’.)"

    return out
