import streamlit as st
import streamlit.components.v1 as components
import google.generativeai as genai
import json
import os
import random
from datetime import datetime, timedelta
from dotenv import load_dotenv
import tempfile
import requests
import base64
import hashlib

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
google_cloud_api_key = os.getenv("GOOGLE_CLOUD_API_KEY")
rubberband_api_key = os.getenv("RUBBERBAND_API_KEY")

# Configure API keys
if not gemini_api_key:
    raise ValueError("No GEMINI_API_KEY found in environment variables.")
if not google_cloud_api_key:
    raise ValueError("No GOOGLE_CLOUD_API_KEY found in environment variables.")
if not rubberband_api_key:
    raise ValueError("No RUBBERBAND_API_KEY found in environment variables.")

genai.configure(api_key=gemini_api_key)

# Define available books and their configurations
BOOKS = {
    "Our Curated Reads": {
        "icon": "📚",
        "format": "[Quote] (Author, Book)",
        "is_religious": False,
        "prayer_label": "Daily Thought",
        "insight_label": "Daily Insight",
        "reflection_label": "Guided Reflection"
    },
    "Bible": {
        "icon": "📖",
        "format": "[Verse text] ([Book Chapter:Verse])",
        "is_religious": True,
        "prayer_label": "Daily Prayer",
        "insight_label": "Religious Insight",
        "reflection_label": "Guided Scripture"
    },
    "Quran": {
        "icon": "☪️",
        "format": "[Verse text] (Surah:Ayah)",
        "is_religious": True,
        "prayer_label": "Daily Prayer",
        "insight_label": "Religious Insight",
        "reflection_label": "Guided Scripture"
    },
    "Bhagavad Gita": {
        "icon": "🕉️",
        "format": "[Verse text] (Chapter.Verse)",
        "is_religious": True,
        "prayer_label": "Daily Prayer",
        "insight_label": "Religious Insight",
        "reflection_label": "Guided Scripture"
    },
    "Torah/Talmud": {
        "icon": "✡️",
        "format": "[Verse text] (Book Chapter:Verse)",
        "is_religious": True,
        "prayer_label": "Daily Prayer",
        "insight_label": "Religious Insight",
        "reflection_label": "Guided Scripture"
    },
    "Dhammapada/Tripitaka": {
        "icon": "☸️",
        "format": "[Verse text] (Chapter.Verse)",
        "is_religious": True,
        "prayer_label": "Daily Prayer",
        "insight_label": "Religious Insight",
        "reflection_label": "Guided Scripture"
    }
}


def get_random_style():
    """Get random style for overlay text"""
    styles = {
        'modern': {
            'font_family': '"Montserrat", sans-serif',
            'text_transform': 'uppercase',
            'letter_spacing': '0.1em',
            'gradient': 'linear-gradient(rgba(0,0,0,0.3), rgba(0,0,0,0.7))',
            'key_phrase_size': '2.5rem',
            'reference_size': '1.25rem'
        },
        'classic': {
            'font_family': '"Playfair Display", serif',
            'text_transform': 'none',
            'letter_spacing': '0.05em',
            'gradient': 'linear-gradient(rgba(0,0,0,0.4), rgba(0,0,0,0.6))',
            'key_phrase_size': '2.75rem',
            'reference_size': '1.5rem'
        },
        'minimalist': {
            'font_family': '"Raleway", sans-serif',
            'text_transform': 'lowercase',
            'letter_spacing': '0.15em',
            'gradient': 'linear-gradient(rgba(0,0,0,0.2), rgba(0,0,0,0.8))',
            'key_phrase_size': '2.25rem',
            'reference_size': '1.1rem'
        },
        'elegant': {
            'font_family': '"Cormorant Garamond", serif',
            'text_transform': 'capitalize',
            'letter_spacing': '0.08em',
            'gradient': 'linear-gradient(rgba(0,0,0,0.5), rgba(0,0,0,0.75))',
            'key_phrase_size': '3rem',
            'reference_size': '1.4rem'
        }
    }
    return random.choice(list(styles.values()))


def create_overlay_html(image_url, key_phrase, reference, style=None):
    """Create HTML for image overlay"""
    if 'overlay_style' not in st.session_state:
        st.session_state.overlay_style = get_random_style()

    style = style or st.session_state.overlay_style

    return f"""
    <html>
    <head>
        <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&family=Playfair+Display:wght@400;700&family=Raleway:wght@400;700&family=Cormorant+Garamond:wght@400;700&display=swap" rel="stylesheet">
        <style>
            .verse-container {{
                position: relative;
                width: 100%;
                height: 400px;
                overflow: hidden;
                border-radius: 12px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            }}
            .verse-image {{
                width: 100%;
                height: 100%;
                object-fit: cover;
            }}
            .verse-overlay {{
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: {style['gradient']};
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                padding: 2rem;
                text-align: center;
            }}
            .key-phrase {{
                color: white;
                font-family: {style['font_family']};
                font-size: {style['key_phrase_size']};
                font-weight: 700;
                text-transform: {style['text_transform']};
                letter-spacing: {style['letter_spacing']};
                text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
                margin-bottom: 1rem;
                max-width: 90%;
                line-height: 1.2;
            }}
            .verse-reference {{
                color: rgba(255,255,255,0.9);
                font-family: {style['font_family']};
                font-size: {style['reference_size']};
                opacity: 0.8;
            }}
        </style>
    </head>
    <body>
        <div class="verse-container">
            <img src="{image_url}" class="verse-image" alt="Verse background"/>
            <div class="verse-overlay">
                <div class="key-phrase">{key_phrase}</div>
                <div class="verse-reference">{reference}</div>
            </div>
        </div>
    </body>
    </html>
    """


def extract_key_phrase(verse, book_config):
    """Extract key phrase from verse for overlay"""
    try:
        spiritual_model = genai.GenerativeModel(
            model_name="gemini-1.5-pro"
        )
        response = spiritual_model.generate_content(
            f"""Extract a short, impactful part from this {book_config['is_religious'] and 'verse' or 'quote'}: {verse}
            This part should be between 4 to 10 words long and should have meaning when read alone.
            It should capture the main message or theme. Return only the extracted part, nothing else."""
        )
        return response.text.strip()
    except:
        # Fallback: take first 5-7 words
        main_text = verse.split('(')[0].strip()
        words = main_text.split()
        return ' '.join(words[:min(7, len(words))])


class ContentTracker:
    def __init__(self):
        self.init_session_state()

    def init_session_state(self):
        """Initialize session state for content tracking"""
        if 'content_history' not in st.session_state:
            st.session_state.content_history = {}
        if 'last_generated' not in st.session_state:
            st.session_state.last_generated = {}

    def get_content_hash(self, content):
        """Generate a unique hash for the content"""
        content_str = f"{content.get('daily_verse', '')}{content.get('guided_scripture', '')}"
        return hashlib.md5(content_str.encode()).hexdigest()

    def extract_reference(self, verse, book):
        """Extract reference from verse text safely"""
        try:
            if '(' not in verse or ')' not in verse:
                return None

            ref_part = verse.split('(')[-1].split(')')[0].strip()

            if ':' in ref_part:  # Bible, Torah, Quran format
                chapter, verse = ref_part.split(':')
                return f"{book}:{chapter}:{verse}"
            elif '.' in ref_part:  # Bhagavad Gita, Dhammapada format
                chapter, verse = ref_part.split('.')
                return f"{book}:{chapter}:{verse}"
            else:  # Other formats
                return f"{book}:{ref_part}"

        except Exception as e:
            st.error(f"Error parsing verse reference: {e}")
            return None

    def is_content_unique(self, content, book):
        """Check if content is unique and not recently used"""
        if not content:
            return False

        content_hash = self.get_content_hash(content)
        verse_ref = self.extract_reference(content.get('daily_verse', ''), book)

        if book not in st.session_state.content_history:
            st.session_state.content_history[book] = {
                'hashes': [],
                'refs': set(),
                'last_cleanup': datetime.now()
            }

        self._cleanup_old_entries(book)

        history = st.session_state.content_history[book]

        if content_hash in history['hashes']:
            return False

        if verse_ref and verse_ref in history['refs']:
            return False

        if self._is_generated_too_recently(book):
            return False

        history['hashes'].append(content_hash)
        if verse_ref:
            history['refs'].add(verse_ref)

        history['hashes'] = history['hashes'][-50:]
        if len(history['refs']) > 30:
            history['refs'].pop()

        st.session_state.last_generated[book] = datetime.now()

        return True

    def _cleanup_old_entries(self, book):
        """Clean up old entries weekly"""
        history = st.session_state.content_history[book]
        last_cleanup = history['last_cleanup']

        if datetime.now() - last_cleanup > timedelta(days=7):
            history['refs'] = set(list(history['refs'])[-30:])
            history['hashes'] = history['hashes'][-50:]
            history['last_cleanup'] = datetime.now()

    def _is_generated_too_recently(self, book):
        """Check if content was generated too recently"""
        if book not in st.session_state.last_generated:
            return False

        last_time = st.session_state.last_generated[book]
        return datetime.now() - last_time < timedelta(seconds=30)


def get_base_prompt(book):
    """Generate a base prompt based on the selected book"""
    book_config = BOOKS[book]

    return f"""
    Generate a JSON object with {book} content using these guidelines:

    daily_verse: 
    - {book} verse/quote formatted as "{book_config['format']}"
    - Choose meaningful, impactful verses/quotes
    - For non-religious texts, select inspiring quotes

    daily_devotional: 
    - 50-word reflection on the verse/quote
    - Focus on practical application and insight

    prayer_guide: 
    - 50-word {'prayer' if book_config['is_religious'] else 'thought'} related to the verse/quote
    - {'Include praise, gratitude, and request elements' if book_config['is_religious'] else 'Include reflection, inspiration, and personal growth elements'}

    insight: 
    - Brief {'religious' if book_config['is_religious'] else 'philosophical'} fact or tradition
    - Relevant to current season or verse/quote theme

    guided_scripture: 
    - 250-word {'sermon' if book_config['is_religious'] else 'reflection'} in engaging, conversational format
    - Structure:
        * Start with a compelling hook
        * Include a relevant contemporary anecdote or story
        * Weave in {'verses' if book_config['is_religious'] else 'quotes'} naturally into the narrative
        * Connect ancient wisdom to modern life
        * End with a powerful, memorable takeaway

    verse_image_prompt: 
    - 30-word image prompt. Don't show human faces
    - Specify art style, colors, and composition
    - Focus on metaphorical elements

    Output Format: Valid JSON with above keys
    """


def generate_image(prompt):
    """Generate image using Rubberband API"""
    try:
        response = requests.post(
            "https://superman.rubbrband.com/get_rubbrband_image",
            json={
                "api_key": rubberband_api_key,
                "prompt": prompt,
                "num_images": 1,
                "aspect_ratio": "16:9"
            }
        )
        response.raise_for_status()
        urls = response.json().get("data", [])
        return urls[0] if urls else None
    except Exception as e:
        st.error(f"Error generating image: {e}")
        return None


def convert_text_to_speech(text):
    """Convert text to audio using Google Cloud Text-to-Speech"""
    try:
        url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={google_cloud_api_key}"
        headers = {'Content-Type': 'application/json'}
        data = {
            'input': {'text': text},
            'voice': {
                'languageCode': 'en-US',
                'name': 'en-US-Neural2-D',
                'ssmlGender': 'MALE'
            },
            'audioConfig': {
                'audioEncoding': 'MP3',
                'speakingRate': 0.85,
                'pitch': 0.0,
                'volumeGainDb': 1.0
            }
        }

        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        audio_content = response.json().get('audioContent')

        if not audio_content:
            raise ValueError("No audio content received from the API")

        audio_data = base64.b64decode(audio_content)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            temp_audio.write(audio_data)
            return temp_audio.name

    except Exception as e:
        st.error(f"Error in text-to-speech conversion: {e}")
        return None


def get_spiritual_content(book):
    """Generate unique spiritual content"""
    content_tracker = ContentTracker()

    spiritual_model = genai.GenerativeModel(
        model_name="gemini-1.5-pro",
        system_instruction=get_base_prompt(book)
    )

    max_attempts = 5

    for attempt in range(max_attempts):
        try:
            prompt = f"""
            Generate unique content for {book}.
            Requirements:
            - Format verse/quote as specified for {book}
            - Ensure content is unique and different from recent generations
            - Focus on {['wisdom', 'guidance', 'reflection', 'inspiration'][attempt % 4]} theme
            - Maintain specified word counts
            - Format guided_scripture as continuous paragraph
            - Make content flow naturally
            """

            response = spiritual_model.generate_content(prompt)
            text = response.parts[0].text.strip()

            if text.startswith('```json'):
                text = text[7:]
            if text.endswith('```'):
                text = text[:-3]

            content = json.loads(text)

            if content_tracker.is_content_unique(content, book):
                return content

        except Exception as e:
            if attempt == max_attempts - 1:
                st.error(f"Error generating unique content: {e}")
                return get_fallback_content(book)

    st.warning("Generated content may be similar to previous versions.")
    return get_fallback_content(book)


def get_fallback_content(book):
    """Provide fallback content if generation fails"""
    is_religious = BOOKS[book]["is_religious"]
    return {
        "daily_verse": f"Life is a journey of discovery and growth ({book})",
        "daily_devotional": "Reflect on the power of personal growth and inner wisdom.",
        "prayer_guide": "May we find strength and guidance in our daily journey.",
        "insight": f"{'Spiritual' if is_religious else 'Personal'} growth comes through daily reflection and practice.",
        "guided_scripture": "Today, let us reflect on the power of personal transformation...",
        "verse_image_prompt": "peaceful landscape, soft light through clouds, watercolor style"
    }


def initialize_session_state():
    """Initialize session state variables"""
    if 'book_selected' not in st.session_state:
        st.session_state.book_selected = False
    if 'current_book' not in st.session_state:
        st.session_state.current_book = None
    if 'content' not in st.session_state:
        st.session_state.content = None
    if 'content_history' not in st.session_state:
        st.session_state.content_history = {}
    if 'last_generated' not in st.session_state:
        st.session_state.last_generated = {}


def main():
    initialize_session_state()

    st.title("Daily Grace")
    st.write(
        "Welcome to Daily Grace! Discover inspiration, comfort, and guidance every day. Choose a book to be inspired from and let us provide you with meaningful reflections and insights.")

    if not st.session_state.book_selected:
        selected_book = st.selectbox(
            "Choose your source of inspiration:",
            list(BOOKS.keys()),
            format_func=lambda x: f"{BOOKS[x]['icon']} {x}"
        )

        if st.button("Get Started"):
            st.session_state.book_selected = True
            st.session_state.current_book = selected_book
            st.rerun()

    else:
        with st.sidebar:
            st.title("📚 Customize Your Experience")
            new_book = st.selectbox(
                "Change your book of inspiration:",
                list(BOOKS.keys()),
                index=list(BOOKS.keys()).index(st.session_state.current_book),
                format_func=lambda x: f"{BOOKS[x]['icon']} {x}"
            )
            if new_book != st.session_state.current_book:
                st.session_state.current_book = new_book
                st.session_state.content = None
                st.rerun()

        book_config = BOOKS[st.session_state.current_book]

        st.subheader(f"{book_config['icon']} Today's Inspiration from {st.session_state.current_book}")

        if st.button("🔄 Generate New Content") or not st.session_state.content:
            with st.spinner("Generating content..."):
                st.session_state.content = get_spiritual_content(st.session_state.current_book)
                if "verse_image_prompt" in st.session_state.content:
                    with st.spinner("Creating verse illustration..."):
                        image_url = generate_image(st.session_state.content["verse_image_prompt"])
                        if image_url:
                            st.session_state.image_url = image_url
                st.rerun()

        st.markdown(f"### 📖 Daily {book_config['is_religious'] and 'Verse' or 'Quote'}")
        verse = st.session_state.content.get('daily_verse', 'No verse available.')
        st.markdown(
            f"<div style='font-size: 24px; padding: 20px 0;'>{verse}</div>",
            unsafe_allow_html=True)

        if 'image_url' in st.session_state:
            key_phrase = extract_key_phrase(verse, book_config)
            reference = verse.split('(')[-1].split(')')[0].strip() if '(' in verse else ''

            overlay_html = create_overlay_html(
                st.session_state.image_url,
                key_phrase,
                reference
            )
            components.html(overlay_html, height=450)

        st.subheader("💭 Daily Reflection")
        st.write(st.session_state.content.get("daily_devotional", "Today's reflection could not be generated."))

        st.subheader(f"🙏 {book_config['prayer_label']}")
        st.write(st.session_state.content.get("prayer_guide", "A simple prayer for guidance."))

        st.subheader(f"🕊️ {book_config['insight_label']}")
        st.write(st.session_state.content.get("insight", "Each day brings new wisdom."))

        st.subheader(f"🎙️ {book_config['reflection_label']}")
        if "guided_scripture" in st.session_state.content:
            reflection_text = st.session_state.content["guided_scripture"]
            with st.expander("Read Reflection", expanded=True):
                st.write(reflection_text)

            col1, col2 = st.columns([1, 2])
            with col1:
                if st.button("🔊 Generate Audio"):
                    with st.spinner("Converting to speech..."):
                        audio_file = convert_text_to_speech(reflection_text)
                        if audio_file:
                            st.session_state.audio_file = audio_file
                            st.session_state.show_download = True

            with col2:
                if 'audio_file' in st.session_state and 'show_download' in st.session_state:
                    st.audio(st.session_state.audio_file, format='audio/mp3')
                    st.download_button(
                        label="📥 Download Audio",
                        data=open(st.session_state.audio_file, 'rb'),
                        file_name=f"daily_reflection_{st.session_state.current_book.lower().replace('/', '_')}.mp3",
                        mime="audio/mp3"
                    )


if __name__ == "__main__":
    main()