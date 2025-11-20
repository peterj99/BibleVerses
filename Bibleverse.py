import streamlit as st
import streamlit.components.v1 as components
import google.generativeai as genai
from google.cloud import texttospeech
from google.oauth2 import service_account
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
import json
import re
import base64
import tempfile

# --- Configuration & Auth ---
st.set_page_config(page_title="Daily Grace", page_icon="🙏", layout="centered")

# Load Secrets
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    PROJECT_ID = st.secrets["GCP_PROJECT_ID"]
    LOCATION = st.secrets["GCP_LOCATION"]
    # Load GCP Credentials from secrets TOML dictionary
    GCP_CREDS = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"]
    )
except Exception as e:
    st.error(f"❌ Missing Secrets: {e}. Please check .streamlit/secrets.toml")
    st.stop()

# Initialize Clients
genai.configure(api_key=GEMINI_API_KEY)
vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=GCP_CREDS)

# --- Class Definitions ---

class CacheManager:
    """Manages session state to persist data across reruns"""
    def __init__(self):
        if 'content_cache' not in st.session_state:
            st.session_state.content_cache = {}
        if 'current_content' not in st.session_state:
            st.session_state.current_content = None
        if 'audio_data' not in st.session_state:
            st.session_state.audio_data = None
        if 'generated_image_url' not in st.session_state:
            st.session_state.generated_image_url = None

    def clear_current(self):
        st.session_state.generated_image_url = None
        st.session_state.audio_data = None


class ContentGenerator:
    """Handles Text Generation via Gemini"""
    def __init__(self):
        # Using the modern model with native JSON support
        self.model = genai.GenerativeModel(
            "gemini-1.5-flash", 
            generation_config={"response_mime_type": "application/json"}
        )

    def get_spiritual_content(self):
        prompt = """
        You are a compassionate spiritual guide. Generate a JSON object containing:
        1. "daily_verse": A Bible verse (KJV) with reference.
        2. "key_phrase": A 3-5 word impactful excerpt from the verse.
        3. "daily_devotional": A 60-word encouraging reflection.
        4. "prayer_guide": A 50-word prayer.
        5. "religious_insight": A fascinating historical or context fact about the verse.
        6. "guided_scripture_sermon": A 200-word warm, spoken-word style mini-sermon suitable for audio.
        7. "image_prompt": A high-quality prompt for an AI image generator. 
           - Style: Cinematic, ethereal, nature-focused, golden hour lighting.
           - Subject: Symbolic representation of the verse. 
           - Constraint: NO TEXT, NO HUMAN FACES. 
        
        Ensure the JSON keys match exactly.
        """
        try:
            response = self.model.generate_content(prompt)
            return json.loads(response.text)
        except Exception as e:
            st.error(f"Error generating text: {e}")
            return None


class ImageGenerator:
    """Handles Image Generation via Vertex AI (Imagen)"""
    def generate(self, prompt):
        try:
            # Using Imagen 3 (Preview) or Imagen 2
            model = ImageGenerationModel.from_pretrained("imagegeneration@006") 
            
            images = model.generate_images(
                prompt=prompt,
                number_of_images=1,
                aspect_ratio="16:9",
                safety_filter_level="block_some",
                person_generation="allow_adult" 
            )
            
            # Save to temp file to display
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_img:
                images[0].save(temp_img.name)
                return temp_img.name
                
        except Exception as e:
            # Fallback if Vertex is not enabled
            st.warning(f"Vertex AI Image Gen failed ({e}). Using placeholder.")
            return "https://images.unsplash.com/photo-1507692049790-de58293a469d?q=80&w=1000&auto=format&fit=crop"


class AudioGenerator:
    """Handles Text-to-Speech via Google Cloud Client Library"""
    def __init__(self, credentials):
        self.client = texttospeech.TextToSpeechClient(credentials=credentials)

    def generate_speech(self, text):
        synthesis_input = texttospeech.SynthesisInput(text=text)

        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Journey-D", # "Journey" voices are much more human-like
            ssml_gender=texttospeech.SsmlVoiceGender.MALE
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=0.90, # Slightly slower for gravitas
            pitch=-1.0
        )

        try:
            response = self.client.synthesize_speech(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )
            return response.audio_content
        except Exception as e:
            st.error(f"TTS Error: {e}")
            return None


# --- UI Components ---

def render_overlay(image_path, phrase, reference):
    """Renders the HTML/CSS overlay for the verse"""
    
    # If it's a local file, we need to encode it
    if image_path.startswith("http"):
        img_src = image_path
    else:
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        img_src = f"data:image/png;base64,{encoded}"

    html_code = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@700&family=Lato:wght@400&display=swap');
        .container {{
            position: relative; width: 100%; height: 400px; border-radius: 15px; overflow: hidden;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        .bg-img {{
            width: 100%; height: 100%; object-fit: cover;
            transition: transform 5s ease;
        }}
        .container:hover .bg-img {{ transform: scale(1.05); }}
        .overlay {{
            position: absolute; bottom: 0; left: 0; width: 100%; height: 60%;
            background: linear-gradient(to top, rgba(0,0,0,0.9), transparent);
            display: flex; flex-direction: column; justify-content: flex-end;
            padding: 30px; box-sizing: border-box; text-align: center;
        }}
        .phrase {{
            font-family: 'Cinzel', serif; color: #fff; font-size: 28px; 
            text-transform: uppercase; letter-spacing: 2px; margin-bottom: 10px;
            text-shadow: 0 2px 4px rgba(0,0,0,0.5);
        }}
        .ref {{
            font-family: 'Lato', sans-serif; color: rgba(255,255,255,0.8); font-size: 16px; font-style: italic;
        }}
    </style>
    <div class="container">
        <img src="{img_src}" class="bg-img">
        <div class="overlay">
            <div class="phrase">{phrase}</div>
            <div class="ref">{reference}</div>
        </div>
    </div>
    """
    components.html(html_code, height=420)


# --- Main App Logic ---

def main():
    cache = CacheManager()
    content_gen = ContentGenerator()
    img_gen = ImageGenerator()
    audio_gen = AudioGenerator(GCP_CREDS)

    st.title("Daily Grace 🌿")
    st.markdown("_Inspiration, comfort, and guidance powered by Enterprise AI_")
    st.divider()

    # Generate Button
    col_act, col_info = st.columns([1, 3])
    with col_act:
        if st.button("✨ New Devotion", type="primary", use_container_width=True):
            cache.clear_current()
            with st.spinner("Consulting scripture..."):
                st.session_state.current_content = content_gen.get_spiritual_content()
            
            if st.session_state.current_content:
                with st.spinner("Painting vision..."):
                    prompt = st.session_state.current_content.get("image_prompt", "peaceful nature")
                    st.session_state.generated_image_url = img_gen.generate(prompt)

    # Display Content
    content = st.session_state.current_content
    
    if content:
        # 1. Visual Header
        render_overlay(
            st.session_state.generated_image_url, 
            content.get("key_phrase", "Grace"), 
            content.get("daily_verse", "").split("(")[-1].strip(")") 
        )

        # 2. The Verse
        st.info(f"**Scripture:** {content.get('daily_verse')}")

        # 3. Three Column Layout
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("### 💭 Reflection")
            st.write(content.get("daily_devotional"))
        with c2:
            st.markdown("### 🙏 Prayer")
            st.write(content.get("prayer_guide"))
        with c3:
            st.markdown("### 🕯️ Insight")
            st.write(content.get("religious_insight"))

        st.divider()

        # 4. Audio Section
        st.markdown("### 🎙️ Guided Meditation")
        sermon_text = content.get("guided_scripture_sermon")
        
        with st.expander("Read the message"):
            st.write(sermon_text)

        if st.button("🔊 Listen to Message"):
            if not st.session_state.audio_data:
                with st.spinner("Synthesizing voice..."):
                    st.session_state.audio_data = audio_gen.generate_speech(sermon_text)
            
            if st.session_state.audio_data:
                st.audio(st.session_state.audio_data, format="audio/mp3")

if __name__ == "__main__":
    main()
