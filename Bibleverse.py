import streamlit as st
import google.generativeai as genai
from google.cloud import texttospeech
from google.oauth2 import service_account
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
import json
import datetime
import random
import hashlib
from typing import Optional, List, Tuple

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Luminary", page_icon="🕯️", layout="centered")

# Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&family=Lato:wght@300;400;700&display=swap');
    
    h1, h2, h3, h4 { font-family: 'Cinzel', serif; color: #1a1a1a; }
    p, div, button, span { font-family: 'Lato', sans-serif; color: #333; }
    
    .daily-card {
        padding: 20px;
        background: #f8f9fa;
        border-radius: 15px;
        border-left: 5px solid #333;
        margin-bottom: 30px;
    }
    
    .pivot-col-header {
        font-size: 14px;
        font-weight: 700;
        text-transform: uppercase;
        color: #666;
        text-align: center;
        min-height: 40px;
        margin-bottom: 10px;
    }
    
    .stButton button {
        width: 100%;
        border-radius: 8px;
        border: 1px solid #eee;
        background-color: white;
        transition: all 0.2s;
    }
    .stButton button:hover {
        border-color: #333;
        background-color: #f4f4f4;
    }
    </style>
""", unsafe_allow_html=True)

# Auth Setup
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    GCP_CREDS = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])
    genai.configure(api_key=GEMINI_API_KEY)
    tts_client = texttospeech.TextToSpeechClient(credentials=GCP_CREDS)
    vertexai.init(project=st.secrets["GCP_PROJECT_ID"], location=st.secrets["GCP_LOCATION"], credentials=GCP_CREDS)
except Exception as e:
    st.error(f"Configuration error: {e}")
    st.stop()

# --- 2. DATA DEFINITIONS ---

BOOKS = {
    "Universal": {"icon": "🌍", "desc": "Wisdom", "source_type": "secular philosophy, Stoicism, modern wisdom"},
    "Bible": {"icon": "📖", "desc": "Christian", "source_type": "Biblical scripture"},
    "Quran": {"icon": "☪️", "desc": "Islamic", "source_type": "Quranic verses"},
    "Bhagavad Gita": {"icon": "🕉️", "desc": "Vedic", "source_type": "Bhagavad Gita wisdom"},
    "Torah": {"icon": "✡️", "desc": "Jewish", "source_type": "Torah teachings"}
}

# --- 3. CONTENT TRACKING ---

class ContentTracker:
    """Tracks content history to ensure freshness"""
    
    def __init__(self):
        if 'content_history' not in st.session_state:
            st.session_state.content_history = []
        if 'session_count' not in st.session_state:
            st.session_state.session_count = 0
    
    def add_theme(self, theme: str):
        """Add a theme to history"""
        st.session_state.content_history.append(theme.lower())
        # Keep only last 3
        st.session_state.content_history = st.session_state.content_history[-3:]
    
    def increment_session(self):
        """Increment session counter and clear if needed"""
        st.session_state.session_count += 1
        if st.session_state.session_count >= 10:
            st.session_state.content_history = []
            st.session_state.session_count = 0
    
    def get_recent_themes(self) -> List[str]:
        """Get recent themes to avoid"""
        return st.session_state.content_history

# --- 4. TIME CONTEXT ---

def get_time_context() -> dict:
    """Get current time context for generation"""
    now = datetime.datetime.now()
    hour = now.hour
    day_name = now.strftime("%A")
    
    # Time of day classification
    if 5 <= hour < 12:
        time_period = "morning"
        tone_guide = "gentle, hopeful, energizing for the day ahead"
    elif 12 <= hour < 17:
        time_period = "afternoon"
        tone_guide = "supportive, encouraging, helping push through the day"
    elif 17 <= hour < 21:
        time_period = "evening"
        tone_guide = "reflective, calming, helping process the day"
    else:
        time_period = "night"
        tone_guide = "peaceful, soothing, preparing for rest"
    
    return {
        "time_period": time_period,
        "hour": hour,
        "day_name": day_name,
        "tone_guide": tone_guide
    }

# --- 5. STRUCTURED OUTPUT SCHEMAS ---

CONTENT_SCHEMA = genai.protos.Schema(
    type=genai.protos.Type.OBJECT,
    properties={
        "wisdom_bridge": genai.protos.Schema(type=genai.protos.Type.STRING),
        "source_text": genai.protos.Schema(type=genai.protos.Type.STRING),
        "source_reference": genai.protos.Schema(type=genai.protos.Type.STRING),
        "spoken_sermon": genai.protos.Schema(type=genai.protos.Type.STRING),
        "action_step": genai.protos.Schema(type=genai.protos.Type.STRING),
        "image_prompt": genai.protos.Schema(type=genai.protos.Type.STRING),
        "theme": genai.protos.Schema(type=genai.protos.Type.STRING),
        "shareable_quote": genai.protos.Schema(type=genai.protos.Type.STRING),
    },
    required=["wisdom_bridge", "source_text", "source_reference", "spoken_sermon", "action_step", "image_prompt", "theme", "shareable_quote"]
)

MENU_SCHEMA = genai.protos.Schema(
    type=genai.protos.Type.OBJECT,
    properties={
        "column1": genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                "header": genai.protos.Schema(type=genai.protos.Type.STRING),
                "buttons": genai.protos.Schema(
                    type=genai.protos.Type.ARRAY,
                    items=genai.protos.Schema(
                        type=genai.protos.Type.OBJECT,
                        properties={
                            "label": genai.protos.Schema(type=genai.protos.Type.STRING),
                            "theme": genai.protos.Schema(type=genai.protos.Type.STRING),
                            "context": genai.protos.Schema(type=genai.protos.Type.STRING),
                        }
                    )
                )
            }
        ),
        "column2": genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                "header": genai.protos.Schema(type=genai.protos.Type.STRING),
                "buttons": genai.protos.Schema(
                    type=genai.protos.Type.ARRAY,
                    items=genai.protos.Schema(
                        type=genai.protos.Type.OBJECT,
                        properties={
                            "label": genai.protos.Schema(type=genai.protos.Type.STRING),
                            "theme": genai.protos.Schema(type=genai.protos.Type.STRING),
                            "context": genai.protos.Schema(type=genai.protos.Type.STRING),
                        }
                    )
                )
            }
        ),
        "column3": genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                "header": genai.protos.Schema(type=genai.protos.Type.STRING),
                "buttons": genai.protos.Schema(
                    type=genai.protos.Type.ARRAY,
                    items=genai.protos.Schema(
                        type=genai.protos.Type.OBJECT,
                        properties={
                            "label": genai.protos.Schema(type=genai.protos.Type.STRING),
                            "theme": genai.protos.Schema(type=genai.protos.Type.STRING),
                            "context": genai.protos.Schema(type=genai.protos.Type.STRING),
                        }
                    )
                )
            }
        )
    }
)

# --- 6. GENERATORS ---

class ContentGenerator:
    def __init__(self):
        self.model = genai.GenerativeModel(
            "gemini-2.5-flash"
        )
    
    def _build_base_system_instruction(self, book: str, time_ctx: dict) -> str:
        """Build the base system instruction for the wise friend persona"""
        return f"""You are a wise, mature, trusted friend - someone older and more experienced than the user.
You have lived through struggles and come out stronger. The user genuinely believes in you and respects your perspective.

PERSONA RULES:
- Speak with warmth but also gravitas - you've earned your wisdom
- Validate struggles honestly (no toxic positivity)
- Use contemporary language and modern references naturally
- Address the user directly ("you've been feeling...")
- NEVER use first-person references about yourself (no "I remember", "me too", "I've been there")
- Keep tone appropriate for {time_ctx['time_period']} - {time_ctx['tone_guide']}

WRITING STYLE:
- Conversational but not overly casual - you're a trusted advisor
- Use contractions naturally (you're, don't, it's)
- Include brief pauses in speech (..., —)
- Occasionally use natural speech patterns like "Look," "Here's the thing," "You know what helps?"
- Modern references: work-from-home, social media, notifications, screen time, contemporary workplace

SOURCE: Draw from {BOOKS[book]['source_type']}"""

    def generate(self, book: str, theme: str, context: str, avoid_themes: List[str], variation_seed: int) -> Optional[dict]:
        time_ctx = get_time_context()
        base_instruction = self._build_base_system_instruction(book, time_ctx)
        
        # Build avoidance instruction
        avoid_str = ""
        if avoid_themes:
            avoid_str = f"\n\nIMPORTANT: DO NOT use these recently covered themes or concepts: {', '.join(avoid_themes)}"
        
        prompt = f"""Generate wisdom content for this situation:

CONTEXT: {context}
THEME: {theme}
TIME: {time_ctx['day_name']} {time_ctx['time_period']}
VARIATION SEED: {variation_seed} (use this to create a completely different angle than usual)
{avoid_str}

OUTPUT REQUIREMENTS:

1. wisdom_bridge (1 sentence):
   - A relatable opening that validates their struggle
   - Connects to something they're actually feeling
   - NO first-person from you as advisor
   - Example: "When the inbox never stops and every message feels urgent, it's like drowning in quicksand."

2. source_text:
   - The actual quote/verse from {BOOKS[book]['source_type']}
   - Must be authentic and accurate

3. source_reference:
   - Precise citation (e.g., "Marcus Aurelius, Meditations 4.3" or "Proverbs 3:5-6")

4. spoken_sermon (100 words):
   - Natural conversational tone with contractions
   - Include occasional "umm", "...", or natural pauses
   - Start by validating their specific struggle: "{context}"
   - Include a brief, hopeful real-world scenario that mirrors their situation
   - Use modern references where natural
   - End with perspective from your mature wisdom
   - Write for SPOKEN audio delivery (natural rhythm, not formal prose)

5. action_step:
   - ONE specific thing they can do in the next hour
   - Practical and achievable, not generic
   - Example: "Text one person who makes you feel seen - just 'thinking of you'"

6. image_prompt:
   - Create an image that generates HOPE and releases endorphins
   - Use symbolic objects/metaphors, not generic landscapes
   - Match emotional tone: {"bright, uplifting, warm colors" if theme in ["joy", "hope", "courage"] else "gentle, soothing, transitional imagery"}
   - NO text, NO human faces
   - Examples of good prompts:
     * "Golden light breaking through storm clouds onto a single blooming flower in cracked concrete"
     * "Warm coffee steam rising in morning sunlight through window blinds, desk with closed laptop"
     * "Broken chains glowing with golden light, floating particles, dark background transforming to light"
     * "Paper boat navigating calm waters after storm, origami crane flying above, pastel dawn colors"
   - 30-40 words, cinematic photography style

7. theme:
   - Single word theme of this content (e.g., "resilience", "courage", "rest")
   - Used for tracking uniqueness

8. shareable_quote:
   - A short, powerful phrase (under 25 characters) for the image overlay
   - Must be inspiring, memorable, and highly shareable
   - Distill the core message of the source_text into a modern, punchy quote
   - Examples: "Find rest, not rust", "Breathe through it", "Your strength is here"

CRITICAL: Make this feel like guidance from a trusted friend who genuinely understands modern life struggles."""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=CONTENT_SCHEMA
                )
            )
            return json.loads(response.text)
        except Exception as e:
            st.error(f"Content generation error: {e}")
            return None


class MenuGenerator:
    def __init__(self):
        self.model = genai.GenerativeModel(
            "gemini-2.5-flash"
        )

    def generate(self) -> Optional[dict]:
        time_ctx = get_time_context()
        
        prompt = f"""Generate a contextual 3-column pivot menu for a wisdom app.

CURRENT CONTEXT:
- Time: {time_ctx['day_name']} {time_ctx['time_period']} ({time_ctx['hour']}:00)
- User mood: Likely experiencing {time_ctx['tone_guide']}

Create 3 columns with SPECIFIC struggles relevant to this exact time:

COLUMN THEMES:
1. Relationships/People (who's affecting them right now)
2. Mental/Emotional burdens (what's weighing on their mind)
3. Energy/Physical state (how their body/spirit feels)

REQUIREMENTS:
- Column headers: Dynamic questions (2 lines max, use <br> for line break)
- 3 buttons per column
- Button labels: 2-3 words, conversational (e.g., "My Manager", "Money Stress", "Can't Sleep")
- Context descriptions: Specific modern struggles

CRITICAL: Do NOT use double quotes (") anywhere in your response. Use single quotes (') if necessary.

CONTEXTUAL EXAMPLES:
Monday morning: "Sunday Scaries", "Week Ahead Dread", "Need Motivation"
Friday evening: "Exhausted", "Wasted Week Guilt", "Need to Decompress"
Late night: "Can't Shut Off", "Tomorrow Anxiety", "Racing Thoughts"

Make struggles SPECIFIC to {time_ctx['day_name']} {time_ctx['time_period']}."""


        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=MENU_SCHEMA
                )
            )
            return json.loads(response.text)
        except Exception as e:
            st.error(f"Menu generation error: {e}")
            return None

class ImageGenerator:
    def generate(self, image_prompt: str, text_overlay: str) -> Optional[bytes]:
        """Generates an image using Imagen with a text overlay."""
        try:
            # Construct a prompt that asks for the image and the text overlay
            full_prompt = (
                f"{image_prompt}. "
                f"Subtly embed the word '{text_overlay}' into the image in a stylish, artistic font. "
                "The text should be legible but integrated naturally with the scene."
            )
            
            # Use the Vertex AI Imagen model
            model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")
            images = model.generate_images(
                prompt=full_prompt,
                number_of_images=1,
                aspect_ratio="1:1", # Using 1:1 for the card layout
                safety_filter_level="block_some",
                person_generation="allow_adult"
            )
            return images[0]._image_bytes
        except Exception as e:
            st.error(f"Image generation error: {e}")
            return None

class AudioGenerator:
    def __init__(self):
        self.model = genai.GenerativeModel("gemini-2.5-flash-preview-tts")

    def generate(self, text: str, tone_guide: str) -> Optional[bytes]:
        """Generates audio using the native Gemini TTS model for more natural speech."""
        try:
            # Construct a style prompt to guide the AI's delivery
            style_prompt = f"""
            Read this in the voice of a wise, mature, trusted friend.
            Your tone should be {tone_guide}.
            Speak with warmth and gravitas, using natural pauses.
            Your delivery should feel like a personal, comforting message, not a formal speech.

            Here is the text to read:
            "{text}"
            """

            response = self.model.generate_content(
                contents=style_prompt,
                generation_config=genai.GenerationConfig(
                    response_modalities=["AUDIO"],
                    speech_config=genai.types.SpeechConfig(
                        voice_config=genai.types.VoiceConfig(
                            prebuilt_voice_config=genai.types.PrebuiltVoiceConfig(
                                voice_name='Charon', # A deep, informative voice
                            )
                        )
                    ),
                )
            )
            return response.candidates[0].content.parts[0].inline_data.data
        except Exception as e:
            st.error(f"Audio generation error: {e}")
            return None

# --- 7. UI COMPONENTS ---

def render_card(content: dict, audio_bytes: Optional[bytes], image_bytes: Optional[bytes]):
    """Standard Card Display Component"""
    if image_bytes:
        st.image(image_bytes, use_container_width=True)
    
    st.markdown("### 🎙️ The Message")
    if audio_bytes:
        st.audio(audio_bytes, format="audio/mp3")
    
    st.divider()
    st.markdown(f"**{content['wisdom_bridge']}**")
    
    st.markdown(f"""
    <div style="background: #f9f9f9; padding: 20px; border-left: 4px solid #333; margin: 20px 0;">
        <em style="font-size: 18px;">"{content['source_text']}"</em>
        <br><small>— {content['source_reference']}</small>
    </div>
    """, unsafe_allow_html=True)
    
    st.info(f"⚡ **Action:** {content['action_step']}")


def render_pivot_menu(menu_data: dict):
    """The 3-Column Dynamic Menu"""
    st.markdown("---")
    st.subheader("Where else do you need light?")
    st.markdown("<br>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    
    selection = None
    
    with c1:
        st.markdown(f"<div class='pivot-col-header'>{menu_data['column1']['header']}</div>", unsafe_allow_html=True)
        for btn in menu_data['column1']['buttons']:
            if st.button(btn['label'], key=f"c1_{btn['label']}"):
                selection = (btn['theme'], btn['context'])
    
    with c2:
        st.markdown(f"<div class='pivot-col-header'>{menu_data['column2']['header']}</div>", unsafe_allow_html=True)
        for btn in menu_data['column2']['buttons']:
            if st.button(btn['label'], key=f"c2_{btn['label']}"):
                selection = (btn['theme'], btn['context'])
    
    with c3:
        st.markdown(f"<div class='pivot-col-header'>{menu_data['column3']['header']}</div>", unsafe_allow_html=True)
        for btn in menu_data['column3']['buttons']:
            if st.button(btn['label'], key=f"c3_{btn['label']}"):
                selection = (btn['theme'], btn['context'])
    
    return selection

# --- 8. LOCALSTORAGE INTEGRATION ---

def get_stored_book() -> Optional[str]:
    """Get user book from query params (localStorage bridge)"""
    user_book = st.query_params.get("user_book", None)
    if user_book and user_book in BOOKS:
        return user_book
    return None

def save_book_to_storage(book: str):
    """Save user book to localStorage via JavaScript"""
    st.components.v1.html(f"""
        <script>
        localStorage.setItem('user_book', '{book}');
        window.parent.location.href = window.parent.location.pathname + '?user_book={book}';
        </script>
    """, height=0)

# --- 9. MAIN LOGIC ---

def main():
    # Initialize State
    if 'user_book' not in st.session_state:
        stored_book = get_stored_book()
        st.session_state.user_book = stored_book if stored_book else "Universal"
    
    if 'daily_content' not in st.session_state:
        st.session_state.daily_content = None
    if 'daily_audio' not in st.session_state:
        st.session_state.daily_audio = None
    if 'daily_image' not in st.session_state:
        st.session_state.daily_image = None
    if 'pivot_content' not in st.session_state:
        st.session_state.pivot_content = None
    if 'pivot_audio' not in st.session_state:
        st.session_state.pivot_audio = None
    if 'pivot_image' not in st.session_state:
        st.session_state.pivot_image = None
    if 'temp_selection' not in st.session_state:
        st.session_state.temp_selection = None
    if 'show_book_modal' not in st.session_state:
        st.session_state.show_book_modal = False
    if 'menu_data' not in st.session_state:
        st.session_state.menu_data = None
    if 'variation_seed' not in st.session_state:
        st.session_state.variation_seed = random.randint(1000, 9999)

    # Initialize tracker
    tracker = ContentTracker()

    # --- PART 1: THE DAILY ANCHOR ---
    if not st.session_state.daily_content:
        with st.spinner("Preparing your daily light..."):
            gen = ContentGenerator()
            avoid_themes = tracker.get_recent_themes()
            
            st.session_state.daily_content = gen.generate(
                st.session_state.user_book,
                "Daily Anchor",
                "Starting your day with wisdom and perspective",
                avoid_themes,
                st.session_state.variation_seed
            )
            
            if st.session_state.daily_content:
                # Track theme
                tracker.add_theme(st.session_state.daily_content.get('theme', 'general'))
                tracker.increment_session()
                
                # Generate image
                st.session_state.daily_image = ImageGenerator().generate(
                    st.session_state.daily_content['image_prompt'],
                    st.session_state.daily_content['shareable_quote']
                )

                # Generate audio
                st.session_state.daily_audio = AudioGenerator().generate(
                    st.session_state.daily_content['spoken_sermon'],
                    get_time_context()['tone_guide'] # Pass the tone guide
                )

    # Render Daily Card
    book_display = st.session_state.user_book
    st.title(f"🕯️ Your Daily {book_display} Anchor")
    
    if st.session_state.daily_content:
        render_card(st.session_state.daily_content, st.session_state.daily_audio, st.session_state.daily_image)
    else:
        st.error("Unable to generate daily content. Please refresh.")

    # --- PART 2: THE PIVOT SECTION ---
    
    if st.session_state.pivot_content:
        st.markdown("---")
        st.markdown("### 🕯️ Your Guidance")
        render_card(st.session_state.pivot_content, st.session_state.pivot_audio, st.session_state.pivot_image)
        if st.button("✨ Clear & Ask Something Else"):
            st.session_state.pivot_content = None
            st.session_state.pivot_audio = None
            st.session_state.variation_seed = random.randint(1000, 9999)
            st.rerun()
    else:
        # Generate menu if not exists
        if not st.session_state.menu_data:
            with st.spinner("Preparing your guidance menu..."):
                menu_gen = MenuGenerator()
                st.session_state.menu_data = menu_gen.generate()
        
        if st.session_state.menu_data:
            user_selection = render_pivot_menu(st.session_state.menu_data)
            
            if user_selection:
                st.session_state.temp_selection = user_selection
                
                # If user hasn't chosen a book yet (still on Universal), show modal
                if st.session_state.user_book == "Universal":
                    st.session_state.show_book_modal = True
                    st.rerun()
                else:
                    # Generate immediately with their saved book
                    generate_pivot_result(st.session_state.user_book, user_selection)

    # --- PART 3: THE MODAL ---
    if st.session_state.show_book_modal and st.session_state.temp_selection:
        theme, context = st.session_state.temp_selection
        
        st.markdown("---")
        st.markdown("### 🌟 Select Your Foundation")
        st.markdown(f"To guide you on **{context}**, choose your wisdom source:")
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            if st.button("📖 Bible", use_container_width=True):
                finalize_book_choice("Bible")
        
        with col_b:
            if st.button("☪️ Quran", use_container_width=True):
                finalize_book_choice("Quran")
        
        with col_c:
            if st.button("🕉️ Gita", use_container_width=True):
                finalize_book_choice("Bhagavad Gita")
        
        col_d, col_e = st.columns(2)
        
        with col_d:
            if st.button("✡️ Torah", use_container_width=True):
                finalize_book_choice("Torah")
        
        with col_e:
            if st.button("🕊️ Keep Universal", use_container_width=True):
                finalize_book_choice("Universal")


def finalize_book_choice(book: str):
    """Handle book selection and generate pivot"""
    st.session_state.user_book = book
    st.session_state.show_book_modal = False
    
    # Save to localStorage
    save_book_to_storage(book)
    
    # Generate pivot content
    if st.session_state.temp_selection:
        generate_pivot_result(book, st.session_state.temp_selection)


def generate_pivot_result(book: str, selection_data: tuple):
    """Generate pivot content"""
    theme, context = selection_data
    tracker = ContentTracker()
    
    with st.spinner(f"Consulting {book} for guidance..."):
        gen = ContentGenerator()
        avoid_themes = tracker.get_recent_themes()
        
        content = gen.generate(
            book, 
            theme, 
            context, 
            avoid_themes,
            st.session_state.variation_seed
        )
        
        if content:
            # Track this theme
            tracker.add_theme(content.get('theme', theme))
            tracker.increment_session()
            
            # Generate image for the pivot content
            image = ImageGenerator().generate(
                content['image_prompt'],
                content['shareable_quote']
            )

            # Generate audio
            audio = AudioGenerator().generate(
                content['spoken_sermon'],
                get_time_context()['tone_guide'] # Pass the tone guide
            )
            st.session_state.pivot_content = content
            st.session_state.pivot_image = image
            st.session_state.pivot_audio = audio
            
            # New seed for next generation
            st.session_state.variation_seed = random.randint(1000, 9999)
        
        st.rerun()


if __name__ == "__main__":
    main()
