import streamlit as st
import streamlit.components.v1 as components
import google.generativeai as genai
import json
import requests
import re
import os
import random
from datetime import datetime
import base64
import tempfile
from dotenv import load_dotenv

# Configure API keys from streamlit secrets
# load_dotenv()
# gemini_api_key = os.getenv("GEMINI_API_KEY")
# google_cloud_api_key = os.getenv("GOOGLE_CLOUD_API_KEY")
# rubberband_api_key = os.getenv("RUBBERBAND_API_KEY")

# streamlit secrets
gemini_api_key = st.secrets["GEMINI_API_KEY"]
google_cloud_api_key = st.secrets["GOOGLE_CLOUD_API_KEY"]
rubberband_api_key = st.secrets["RUBBERBAND_API_KEY"]

# Validate API keys
for key, name in [(gemini_api_key, "GEMINI_API_KEY"),
                  (google_cloud_api_key, "GOOGLE_CLOUD_API_KEY"),
                  (rubberband_api_key, "RUBBERBAND_API_KEY")]:
    if not key:
        raise ValueError(f"No {name} found in secrets.")


class CacheManager:
    def __init__(self):
        self.initialize_cache()

    def initialize_cache(self):
        """Initialize all cache-related session states"""
        if 'content_cache' not in st.session_state:
            st.session_state.content_cache = {}
        if 'image_cache' not in st.session_state:
            st.session_state.image_cache = {}
        if 'audio_cache' not in st.session_state:
            st.session_state.audio_cache = {}

    def get_cached_content(self, verse):
        """Get cached content for a verse if it exists"""
        return st.session_state.content_cache.get(verse)

    def cache_content(self, verse, content):
        """Cache content for a verse"""
        st.session_state.content_cache[verse] = content

    def get_cached_image(self, prompt):
        """Get cached image URL for a prompt"""
        return st.session_state.image_cache.get(prompt)

    def cache_image(self, prompt, url):
        """Cache image URL for a prompt"""
        st.session_state.image_cache[prompt] = url

    def get_cached_audio(self, text_hash):
        """Get cached audio file for text"""
        return st.session_state.audio_cache.get(text_hash)

    def cache_audio(self, text_hash, file_path):
        """Cache audio file path"""
        st.session_state.audio_cache[text_hash] = file_path


class VerseManager:
    def __init__(self, max_history=5):
        self.max_history = max_history
        self.cache_manager = CacheManager()
        self.api_client = None  # Will be set by DailyGraceApp

    def initialize_session_state(self):
        """Initialize session state for verse tracking if not exists"""
        if 'previous_verses' not in st.session_state:
            st.session_state.previous_verses = []

    def extract_reference(self, verse_text):
        """Safely extract verse reference from full verse text"""
        try:
            if '(' in verse_text and ')' in verse_text:
                return verse_text.split('(')[-1].split(')')[0].strip()
            return verse_text
        except Exception:
            return verse_text

    def is_verse_unique(self, verse_text):
        """Check if verse reference is not in recent history"""
        self.initialize_session_state()

        if not verse_text:
            return False

        reference = self.extract_reference(verse_text)
        return reference not in st.session_state.previous_verses

    def add_verse(self, verse_text):
        """Add verse to history and maintain max size"""
        self.initialize_session_state()

        if not verse_text:
            return

        reference = self.extract_reference(verse_text)
        st.session_state.previous_verses.append(reference)
        if len(st.session_state.previous_verses) > self.max_history:
            st.session_state.previous_verses = st.session_state.previous_verses[-self.max_history:]

    def get_unique_content(self):
        """Get content with a unique verse"""
        self.initialize_session_state()

        max_attempts = 3
        for _ in range(max_attempts):
            try:
                content = self.api_client.generate_spiritual_content(st.session_state.previous_verses)
                if not content or 'daily_verse' not in content:
                    continue

                verse = content.get('daily_verse', '')
                if self.is_verse_unique(verse):
                    self.add_verse(verse)
                    return content

            except Exception as e:
                st.error(f"Error generating unique content: {e}")
                continue

        # Fallback to any content as last resort
        return self.api_client.generate_spiritual_content()


class APIClient:
    def __init__(self, gemini_key, google_cloud_key, rubberband_key):
        self.gemini_key = gemini_key
        self.google_cloud_key = google_cloud_key
        self.rubberband_key = rubberband_key
        genai.configure(api_key=self.gemini_key)
        self.spiritual_model = genai.GenerativeModel("gemini-1.5-pro")
        self.cache_manager = CacheManager()

    def generate_spiritual_content(self, previous_verses=None):
        """Generate spiritual content with deduplication check"""
        prompt = self._build_prompt(previous_verses)
        try:
            response = self.spiritual_model.generate_content(prompt)
            text = response.text.strip()
            if text.startswith('```json'):
                text = text[7:-3]
            return json.loads(text)
        except Exception as e:
            st.error(f"Error generating content: {e}")
            return None

    def generate_image(self, prompt):
        """Generate image with error handling and retries"""
        # Check cache first
        cached_image = self.cache_manager.get_cached_image(prompt)
        if cached_image:
            return cached_image

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    "https://superman.rubbrband.com/get_rubbrband_image",
                    json={
                        "api_key": self.rubberband_key,
                        "prompt": prompt,
                        "num_images": 1,
                        "aspect_ratio": "16:9"
                    }
                )
                response.raise_for_status()
                image_url = response.json().get("data", [None])[0]
                if image_url:
                    self.cache_manager.cache_image(prompt, image_url)
                return image_url
            except Exception as e:
                if attempt == max_retries - 1:
                    st.error(f"Error generating image after {max_retries} attempts: {e}")
                    return None
                continue

    def convert_text_to_speech(self, text):
        """Convert text to speech with caching"""
        text_hash = hash(text)

        # Check cache first
        cached_audio = self.cache_manager.get_cached_audio(text_hash)
        if cached_audio:
            return cached_audio

        try:
            response = requests.post(
                f"https://texttospeech.googleapis.com/v1/text:synthesize?key={self.google_cloud_key}",
                json={
                    'input': {'text': text},
                    'voice': {
                        'languageCode': 'en-US',
                        'name': 'en-US-Neural2-D',
                        'ssmlGender': 'MALE'
                    },
                    'audioConfig': {
                        'audioEncoding': 'MP3',
                        'speakingRate': 0.85,
                        'pitch': -0.5,
                        'volumeGainDb': 1
                    }
                }
            )
            response.raise_for_status()

            audio_content = response.json().get('audioContent')
            if not audio_content:
                raise ValueError("No audio content received")

            audio_data = base64.b64decode(audio_content)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
                temp_audio.write(audio_data)
                self.cache_manager.cache_audio(text_hash, temp_audio.name)
                return temp_audio.name

        except Exception as e:
            st.error(f"Text-to-speech error: {e}")
            return None

    def extract_key_phrase(self, verse):
        """Extract key phrase from verse"""
        try:
            response = self.spiritual_model.generate_content(
                f"""Extract a short, impactful part from this verse: {verse}, that captures its essence. 
                This part should be between 4 to 10 words long and should have meaning when read alone. 
                It should convey the main message or theme of the verse. Return only the part, nothing else."""
            )
            return response.text.strip()
        except:
            # Fallback: take first 5 words
            main_verse = verse.split('(')[0].strip()
            return ' '.join(main_verse.split()[:5])

    def _build_prompt(self, previous_verses=None):
        """Build prompt with previous verses for deduplication"""
        previous_verses_text = ""
        if previous_verses:
            previous_verses_text = f"\nAvoid these verses: {', '.join(previous_verses)}"

        return f"""Consider that you are an expert in giving christian religious content. Generate a JSON object with:
        - daily_verse: KJV Bible verse '[Text] (Book Ch:Verse)'{previous_verses_text}
        - daily_devotional: 50-word reflection based on the verse
        - prayer_guide: 50-word prayer based on the verse
        - religious_insight: Brief Christian fact based on the verse which is interesting and not so common.
        - guided_scripture: 250-word sermon based on the verse in an engaging, conversational format. 
            Structure:
            * Start with a compelling hook
            * Weave in verses naturally
            * Connect ancient wisdom to modern life
            * End with a powerful takeaway
            Engagement elements:
            * Use vivid descriptions and imagery
            * Include rhetorical questions
            * Share relatable examples
            * Add appropriate cultural references
            Technical requirements:
            * Flow naturally as spoken content
            * Use full spoken verse format
        - verse_image_prompt: 30-word image prompt. The image should be used for a christian setting. Specify art style, colors, and composition. Avoid human faces.
        Return only valid JSON."""


class DailyGraceApp:
    def __init__(self):
        self.api_client = APIClient(
            gemini_api_key,
            google_cloud_api_key,
            rubberband_api_key
        )
        self.verse_manager = VerseManager(max_history=5)
        self.cache_manager = CacheManager()
        self.verse_manager.api_client = self.api_client

    def get_random_style(self):
        """Get random style for overlay text"""
        styles = {
            'modern': {
                'font_family': '"Montserrat", sans-serif',
                'text_transform': 'uppercase',
                'letter_spacing': '0.1em',
                'gradient': 'linear-gradient(rgba(0,0,0,0.3), rgba(0,0,0,0.7))'
            },
            'classic': {
                'font_family': '"Playfair Display", serif',
                'text_transform': 'none',
                'letter_spacing': '0.05em',
                'gradient': 'linear-gradient(rgba(0,0,0,0.4), rgba(0,0,0,0.6))'
            }
        }
        return random.choice(list(styles.values()))

    def create_overlay_html(self, image_url, key_phrase, reference, style=None):
        """Create HTML for image overlay"""
        # Initialize style in session state if not present
        if 'overlay_style' not in st.session_state:
            st.session_state.overlay_style = self.get_random_style()

        # Use stored style or passed style
        style = style or st.session_state.overlay_style

        return f"""
        <html>
        <head>
            <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&family=Playfair+Display:wght@400;700&display=swap" rel="stylesheet">
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
                    font-size: 2.5rem;
                    font-weight: 700;
                    text-transform: {style['text_transform']};
                    letter-spacing: {style['letter_spacing']};
                    text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
                    margin-bottom: 1rem;
                }}
                .verse-reference {{
                    color: rgba(255,255,255,0.9);
                    font-family: {style['font_family']};
                    font-size: 1.25rem;
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

    def generate_new_content(self):
        """Generate new content with deduplication"""
        content = self.verse_manager.get_unique_content()
        if content:
            st.session_state.content = content
            # Reset the overlay style when generating new content
            st.session_state.overlay_style = self.get_random_style()

            if "verse_image_prompt" in content:
                with st.spinner("Creating verse illustration..."):
                    image_url = self.api_client.generate_image(content["verse_image_prompt"])
                    if image_url:
                        st.session_state.image_url = image_url
                        return True
            return False


def main():
    st.title("Daily Grace")
    st.write("Discover inspiration, comfort, and guidance every day with Daily Grace.")

    app = DailyGraceApp()

    # Generate new content
    if st.button("🔄 Generate New Content") or 'content' not in st.session_state:
        with st.spinner("Generating spiritual content..."):
            if app.generate_new_content():
                st.rerun()

    if 'content' in st.session_state:
        content = st.session_state.content

        # Display verse and overlay image
        st.subheader("📖 Daily Bible Verse")
        verse = content.get("daily_verse", "")
        st.write(verse)

        if 'image_url' in st.session_state:
            key_phrase = app.api_client.extract_key_phrase(verse)
            reference = re.search(r'\((.*?)\)', verse).group(1) if '(' in verse else ''

            overlay_html = app.create_overlay_html(
                st.session_state.image_url,
                key_phrase,
                reference
            )
            components.html(overlay_html, height=450)

        # Display other content
        st.subheader("💭 Daily Devotional")
        st.write(content.get("daily_devotional", ""))

        st.subheader("🙏 Prayer Guide")
        st.write(content.get("prayer_guide", ""))

        st.subheader("🕊️ Religious Insight")
        st.write(content.get("religious_insight", ""))

        # Handle sermon and audio
        st.subheader("🎙️ Guided Scripture")
        sermon_text = content.get("guided_scripture", "")
        if sermon_text:
            with st.expander("Read Sermon", expanded=True):
                st.write(sermon_text)

            col1, col2 = st.columns([1, 2])
            with col1:
                if st.button("🔊 Generate Audio"):
                    with st.spinner("Converting sermon to speech..."):
                        audio_file = app.api_client.convert_text_to_speech(sermon_text)
                        if audio_file:
                            st.session_state.audio_file = audio_file
                            st.session_state.show_download = True

            with col2:
                if 'audio_file' in st.session_state and 'show_download' in st.session_state:
                    st.audio(st.session_state.audio_file)
                    st.download_button(
                        "📥 Download Sermon Audio",
                        data=open(st.session_state.audio_file, 'rb'),
                        file_name="daily_sermon.mp3",
                        mime="audio/mp3"
                    )


if __name__ == "__main__":
    main()