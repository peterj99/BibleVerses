import google.generativeai as genai
import streamlit as st
import json
import os
from datetime import datetime
from dotenv import load_dotenv
import tempfile
import requests
import tempfile
import base64
import requests

# Load environment variables
# load_dotenv()
# gemini_api_key = os.getenv("GEMINI_API_KEY")
# google_cloud_api_key = os.getenv("GOOGLE_CLOUD_API_KEY")
# rubberband_api_key = os.getenv("RUBBERBAND_API_KEY")

# streamlit secrets
gemini_api_key = st.secrets["GEMINI_API_KEY"]
google_cloud_api_key = st.secrets["GOOGLE_CLOUD_API_KEY"]
rubberband_api_key = st.secrets["RUBBERBAND_API_KEY"]


# Configure API keys
if not gemini_api_key:
    raise ValueError("No GEMINI_API_KEY found in environment variables.")
if not google_cloud_api_key:
    raise ValueError("No GOOGLE_CLOUD_API_KEY found in environment variables.")
if not rubberband_api_key:
    raise ValueError("No RUBBERBAND_API_KEY found in environment variables.")

genai.configure(api_key=gemini_api_key)

# Optimized base prompt
base_prompt = """
Generate a JSON object with Christian spiritual content using these guidelines:

daily_verse: 
- KJV Bible verse formatted as "[Verse text] ([Book Chapter:Verse])"
- Choose meaningful, impactful verses

daily_devotional: 
- 80-word reflection on the verse
- Focus on practical application and spiritual insight

prayer_guide: 
- 80-word prayer related to the verse
- Include praise, gratitude, and request elements

religious_insight: 
- Brief Christian fact or tradition
- Relevant to current season or verse theme

guided_scripture: 
- 200-word sermon in plain text format
- Include introduction, message, practical application, and closing thought
- Flow naturally as spoken content
- Include 1-2 supporting verses

verse_image_prompt: 
- 30-word image prompt
- Specify art style, colors, and composition
- Focus on metaphorical elements

Output Format: Valid JSON with above keys
"""

# Initialize the generative model
spiritual_model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    system_instruction=base_prompt
)


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
    """Convert sermon text to audio using Google Cloud Text-to-Speech"""
    try:
        # Make HTTP request to Text-to-Speech API
        url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={google_cloud_api_key}"

        headers = {
            'Content-Type': 'application/json'
        }

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

        # Make the API request
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Extract the audio content from the response
        audio_content = response.json().get('audioContent')

        if not audio_content:
            raise ValueError("No audio content received from the API")

        # Decode the base64 audio content
        import base64
        audio_data = base64.b64decode(audio_content)

        # Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            temp_audio.write(audio_data)
            return temp_audio.name

    except requests.exceptions.RequestException as e:
        error_msg = str(e)
        if "API_KEY_SERVICE_BLOCKED" in error_msg:
            st.error("Please enable the Text-to-Speech API in your Google Cloud Console.")
        elif "API_KEY_INVALID" in error_msg:
            st.error("Invalid Google Cloud API key.")
        else:
            st.error(f"Error in text-to-speech conversion: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error in text-to-speech conversion: {e}")
        return None


def is_verse_unique(new_verse, previous_verses):
    """Check if the verse is unique"""
    try:
        reference = new_verse.split('(')[-1].split(')')[0].strip()
        return reference not in previous_verses
    except:
        return False


def get_spiritual_content(new_verse=False):
    """Generate spiritual content with uniqueness checks"""
    if 'previous_verses' not in st.session_state:
        st.session_state.previous_verses = []

    prompt = f"""
    Generate Christian spiritual content JSON.
    Requirements:
    - KJV verse format: '[Verse text] ([Book Chapter:Verse])'
    - Ensure verse differs from: {', '.join(st.session_state.previous_verses[-5:])}
    - Maintain specified word counts for devotional, prayer, and sermon
    - Format guided_scripture as a continuous paragraph of text, not JSON
    - Make sure the sermon flows naturally as spoken content
    """

    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            response = spiritual_model.generate_content(prompt)
            text = response.parts[0].text.strip()

            if text.startswith('```json'):
                text = text[7:]
            if text.endswith('```'):
                text = text[:-3]

            content = json.loads(text)
            daily_verse = content.get('daily_verse', '')

            if is_verse_unique(daily_verse, st.session_state.previous_verses):
                reference = daily_verse.split('(')[-1].split(')')[0].strip()
                st.session_state.previous_verses.append(reference)
                st.session_state.previous_verses = st.session_state.previous_verses[-5:]
                return content

        except Exception as e:
            if attempt == max_attempts - 1:
                st.error(f"Error generating content: {e}")

    return {
        "daily_verse": "For God so loved the world, that he gave his only begotten Son, that whosoever believeth in him should not perish, but have everlasting life. (John 3:16)",
        "daily_devotional": "Reflect on God's unconditional love and the hope it brings to our lives.",
        "prayer_guide": "Heavenly Father, guide my steps and fill my heart with your peace today.",
        "religious_insight": "The Christian faith is built on the transformative power of love and grace.",
        "guided_scripture": "Today, let us reflect on God's eternal love and its meaning in our lives...",
        "verse_image_prompt": "radiant light through clouds, golden rays illuminating landscape, oil painting, warm gold and blue"
    }


def main():
    st.title("Daily Grace")
    st.write("Discover inspiration, comfort, and guidance every day with Daily Grace.")

    if st.button("🔄 Generate New Content") or 'content' not in st.session_state:
        with st.spinner("Generating spiritual content..."):
            st.session_state.content = get_spiritual_content()
            if "verse_image_prompt" in st.session_state.content:
                with st.spinner("Creating verse illustration..."):
                    image_url = generate_image(st.session_state.content["verse_image_prompt"])
                    if image_url:
                        st.session_state.image_url = image_url
            st.rerun()
        with st.spinner("Generating spiritual content..."):
            st.session_state.content = get_spiritual_content()
            if "verse_image_prompt" in st.session_state.content:
                with st.spinner("Creating verse illustration..."):
                    image_url = generate_image(st.session_state.content["verse_image_prompt"])
                    if image_url:
                        st.session_state.image_url = image_url

    st.subheader("📖 Daily Bible Verse")
    st.write(st.session_state.content.get("daily_verse", "No verse available today."))

    if 'image_url' in st.session_state:
        st.image(st.session_state.image_url, caption="Visual representation of today's verse", use_container_width=True)

    st.subheader("💭 Daily Devotional")
    st.write(st.session_state.content.get("daily_devotional", "Today's devotional could not be generated."))

    st.subheader("🙏 Prayer Guide")
    st.write(st.session_state.content.get("prayer_guide", "A simple prayer for guidance."))

    st.subheader("🕊️ Religious Insight")
    st.write(st.session_state.content.get("religious_insight", "Each day is a gift from God."))

    st.subheader("🎙️ Guided Scripture")
    if "guided_scripture" in st.session_state.content:
        sermon_text = st.session_state.content["guided_scripture"]
        with st.expander("Read Sermon", expanded=True):
            st.write(sermon_text)

        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("🔊 Generate Audio"):
                with st.spinner("Converting sermon to speech..."):
                    audio_file = convert_text_to_speech(sermon_text)
                    if audio_file:
                        st.session_state.audio_file = audio_file
                        st.session_state.show_download = True
                        st.rerun()

        with col2:
            if 'audio_file' in st.session_state and 'show_download' in st.session_state:
                st.audio(st.session_state.audio_file, format='audio/mp3')
                st.download_button(
                    label="📥 Download Sermon Audio",
                    data=open(st.session_state.audio_file, 'rb'),
                    file_name="daily_sermon.mp3",
                    mime="audio/mp3"
                )


if __name__ == "__main__":
    main()