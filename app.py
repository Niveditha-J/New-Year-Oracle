import time
import streamlit as st
from PIL import Image
import predictor
import base64
import io
import random

st.set_page_config(page_title="New Year Oracle", page_icon="✨", layout="centered")

st.markdown("""
<style>
body {background: linear-gradient(180deg,#f8fafc,#ffffff); color: #0b1220}
.card {background: #fff; padding: 18px; border-radius: 14px; box-shadow: 0 6px 20px rgba(11,18,32,0.06);}
.big {font-size:28px; font-weight:800; color:#0b1220}
.small-muted {font-size:12px; color: #6b7280}
.overlay-bubble {background: linear-gradient(90deg,#fff8e1,#fff); border-radius:12px; padding:10px 14px; font-weight:800; color:#0b1220; box-shadow:0 10px 30px rgba(11,18,32,0.08)}
.prediction-chip {display:inline-block; background:#fff1f2; color:#9f1239; padding:10px 12px; border-radius:999px; margin-right:8px; margin-bottom:8px; font-weight:700}
.prediction-primary {background: linear-gradient(90deg,#fff7ed,#ffedd5); border:2px solid #ffd966}
</style>
""", unsafe_allow_html=True)

st.title("New Year Oracle")
st.caption("A playful, entertainment-only vibe scanner for your 2026. For entertainment purposes only.")

# Festive banner
st.markdown("<div style='padding:10px 14px; border-radius:8px; background:linear-gradient(90deg,#fffbeb,#fff7f0); margin-top:6px'><strong>🎄 Merry Christmas — Happy New Year 2026! 🎆</strong></div>", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Step 1 — Take a selfie")
    img_file = st.camera_input("Point your camera and smile (or not)")
    st.write('')
    st.markdown("---")
    st.write('Or upload an image:')
    uploaded_file = st.file_uploader("Upload a photo", type=['png', 'jpg', 'jpeg'])

with col2:
    st.subheader("Optional")
    user_text = st.text_input("How was 2025 for you? (optional)")

# prefer camera capture; if not available use uploaded image
selected_file = img_file if img_file is not None else uploaded_file

if selected_file is not None:
    # Read image bytes so we can embed HTML (allows overlay updates)
    img_bytes = selected_file.getvalue()
    pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

    # quick immediate suggestions overlay (replaced by final results)
    quick_list = predictor.quick_suggestions(25, user_text)

    # build image+overlay HTML and render in a single container so we can update it later
    def make_image_block_bytes(img_b64: str, overlay_html: str) -> str:
        return f"""
        <div style='position:relative; display:inline-block; max-width:100%'>
          <img src='data:image/jpeg;base64,{img_b64}' style='max-width:680px; width:90%; border-radius:12px; display:block;' />
          <div style='position:absolute; top:8px; left:50%; transform:translateX(-50%); background:rgba(255,255,255,0.9); color:#0b1220; padding:10px 14px; border-radius:10px; box-shadow:0 6px 18px rgba(2,6,23,0.25); font-weight:700;'>
            {overlay_html}
          </div>
        </div>
        """

    b64 = base64.b64encode(img_bytes).decode('utf-8')
    placeholder = st.empty()
    # Flicker a single overlay suggestion above the head — quick random picks
    flicker_count = 12
    flicker_delay = 0.12
    # ensure we have a pool to pick from
    pool = quick_list if len(quick_list) > 0 else predictor.quick_suggestions(25, user_text)

    # initialize session set of seen quick titles
    if 'seen_quick_titles' not in st.session_state:
        st.session_state['seen_quick_titles'] = set()

    # prefer showing unseen recommendations (and prioritize non-default ones)
    frequent_three = {
        'Career upgrade detected 💼',
        'New relationship loading… ❤️',
        'Major personal glow-up incoming ✨'
    }

    try:
        for _ in range(flicker_count):
            # choose from unseen non-frequent items first
            unseen = [p for p in pool if p['title'] not in st.session_state['seen_quick_titles']]
            unseen_non_frequent = [p for p in unseen if p['title'] not in frequent_three]

            if unseen_non_frequent:
                pick = random.choice(unseen_non_frequent)
            elif unseen:
                pick = random.choice(unseen)
            else:
                pick = random.choice(pool)

            # mark as seen
            st.session_state['seen_quick_titles'].add(pick['title'])

            overlay_html = f"<div style='font-weight:800; font-size:15px; padding:4px 8px'>{pick['title']}</div>"
            placeholder.markdown(make_image_block_bytes(b64, overlay_html), unsafe_allow_html=True)
            time.sleep(flicker_delay)
    except Exception:
        # if Streamlit prevents quick looping, fall back to a single pick (prefer unseen non-frequent)
        unseen = [p for p in pool if p['title'] not in st.session_state['seen_quick_titles']]
        unseen_non_frequent = [p for p in unseen if p['title'] not in frequent_three]
        if unseen_non_frequent:
            pick = random.choice(unseen_non_frequent)
        elif unseen:
            pick = random.choice(unseen)
        else:
            pick = random.choice(pool)
        st.session_state['seen_quick_titles'].add(pick['title'])
        overlay_html = f"<div style='font-weight:800; font-size:15px; padding:4px 8px'>{pick['title']}</div>"
        placeholder.markdown(make_image_block_bytes(b64, overlay_html), unsafe_allow_html=True)

    # spinner while we do full scan
    with st.spinner('Scanning the vibes and aligning the stars...'):
        time.sleep(0.6)  # small UX pause so the quick overlay registers
        result = predictor.analyze_image(pil_img, user_text)
        time.sleep(0.6)

    # update the image block, picking a primary suggestion that isn't a repeat
    primary = result['predictions'][0]
    # avoid repeating the same primary suggestion twice in a row during the session
    last_primary = st.session_state.get('last_primary')
    titles = [p['title'] for p in result['predictions']]
    if last_primary in titles and len(titles) > 1:
        # prefer a different primary
        for p in result['predictions']:
            if p['title'] != last_primary:
                primary = p
                break

    final_overlay = f"<div class='overlay-bubble'>{primary['title']}</div>"
    placeholder.markdown(make_image_block_bytes(b64, final_overlay), unsafe_allow_html=True)
    st.session_state['last_primary'] = primary['title']

    # show reveal animation beneath the image
    try:
        with open('assets/reveal.svg', 'r', encoding='utf-8') as _f:
            svg = _f.read()
        st.markdown(svg, unsafe_allow_html=True)
    except Exception:
        time.sleep(0.4)

    # show full results below, with primary highlighted
    st.markdown("<div class='card' style='margin-top:14px'>", unsafe_allow_html=True)
    st.markdown(f"<div class='big'>Oracle results — mood: {result['mood'].title()} {result.get('emoji','')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='small-muted'>Confidence: {int(result['confidence']*100)}% — {result.get('note','')}</div>", unsafe_allow_html=True)
    st.markdown("<div style='margin-top:6px; font-style:italic; color:#6b7280'>Wishing you warmth this Christmas and a sparkling 2026.</div>", unsafe_allow_html=True)

    st.write('')
    # ensure final display includes at least one non-default recommendation if available
    display_predictions = result['predictions'][:]
    titles_in_display = {p['title'] for p in display_predictions}
    # candidates come from quick_list (which holds 25 options)
    candidates = [p for p in quick_list if p['title'] not in titles_in_display and p['title'] not in frequent_three]
    if candidates:
        # append one candidate to diversify the final suggestions
        display_predictions.append(random.choice(candidates))

    # show predictions as chips; primary gets the primary style
    chips_html = ''
    for i, p in enumerate(display_predictions):
        cls = 'prediction-chip'
        if p['title'] == primary['title']:
            cls += ' prediction-primary'
        chips_html += f"<span class='{cls}'>{p['title']}</span>"
    st.markdown(chips_html, unsafe_allow_html=True)

    # show details for each (primary first)
    st.write('')
    st.markdown(f"<div style='margin-top:12px'><strong>{primary['title']}</strong><div class='small-muted'>{primary['detail']}</div></div>", unsafe_allow_html=True)
    # other suggestions (show from display_predictions)
    for p in display_predictions:
        if p['title'] == primary['title']:
            continue
        st.markdown(f"<div style='margin-top:8px'><strong>{p['title']}</strong><div class='small-muted'>{p.get('detail','')}</div></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # celebratory flourish based on mood
    if result['mood'] in ('joyful', 'confident', 'playful'):
        st.balloons()
    elif result['mood'] in ('surprised',):
        st.snow()

    st.write('')
    st.markdown("<div class='small-muted'>For entertainment purposes only. This app pretends to predict — it uses simple heuristics and randomized storytelling.</div>", unsafe_allow_html=True)

else:
    st.info("Use your camera or upload an image to let the Oracle scan your vibe.")
