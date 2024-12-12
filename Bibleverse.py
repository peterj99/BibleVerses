import google.generativeai as genai
import streamlit as st
import json
import os
from datetime import datetime

# Configure Gemini API
api_key = st.secrets["GEMINI_API_KEY"]

#test

# Updated Base Prompt with Uniqueness Consideration
base_prompt = """
Objective: You are an expert in creating comprehensive Christian spiritual content. Your task is to generate a JSON object with the following keys: daily_verse, daily_devotional, prayer_guide, and religious_insight.

Guidelines:

daily_verse:
- Generate a unique Bible verse EACH DAY
- Output should be formatted as: "[Verse text] ([Book Chapter:Verse] [Bible Version])"
- Ensure the verse is readable and meaningful
- Confirm the verse is not repetitive
- Use context-appropriate Bible translations

daily_devotional:
- Write a short devotional based on the provided verse (less than 80 words)
- Ensure it's meaningful and theologically sound
- Resonate with a broad Christian audience

prayer_guide:
- Create a prayer based on the verse and devotional content
- Keep it short (less than 100 words)
- Align with Christian traditions

religious_insight:
- Provide a meaningful Christian trivia. It need not be based on the above bible verse or theme
- Can include:
  * Feasts or historical facts about saints that is accepted by all christians
  * Christian historical facts
  * Theological concepts
  * Spiritual practices
  * Biblical interpretations
  * Christian cultural traditions
- If possible, relate the insight to the current day or season in the Christian calendar
- Ensure content is theologically sound and broadly applicable

Output Format: A valid JSON object with unique, formatted content
"""

# Initialize the generative model
spiritual_model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    system_instruction=base_prompt
)


def is_verse_unique(new_verse, previous_verses):
    """
    Check if the verse is unique by comparing its book and verse reference

    Args:
        new_verse (str): The new verse to check
        previous_verses (list): List of previously used verses

    Returns:
        bool: True if the verse is unique, False otherwise
    """
    # Extract book and verse reference
    try:
        # Look for the reference part in parentheses
        reference = new_verse.split('(')[-1].split(')')[0].strip()
        return reference not in previous_verses
    except:
        return False


def get_spiritual_content(bible_version=None, themes=None):
    """
    Generate spiritual content with uniqueness checks

    Args:
        bible_version (str, optional): Preferred Bible version
        themes (list, optional): Spiritual themes

    Returns:
        dict: Spiritual content dictionary
    """
    # Initialize previous verses in session state if not exists
    if 'previous_verses' not in st.session_state:
        st.session_state.previous_verses = []

    # Prepare prompt with version and themes
    prompt = f"""
    Generate comprehensive spiritual content.
    Specific requirements:
    - Bible Version: {bible_version or 'New International Version (NIV)'}
    - Themes: {', '.join(themes or ['general spirituality'])}
    - Ensure verse formatting as '[Verse text] ([Book Chapter:Verse] [Bible Version])'
    - Verify that the verse is UNIQUE from these previous verses: 
      {', '.join(st.session_state.previous_verses[-10:])}
    """

    max_attempts = 3  # Limit attempts to find a unique verse
    for attempt in range(max_attempts):
        try:
            # Generate content
            response = spiritual_model.generate_content(prompt)

            # Extract and parse the JSON content
            text = response.parts[0].text.strip()

            # Remove code block markers if present
            if text.startswith('```json'):
                text = text[7:]
            if text.endswith('```'):
                text = text[:-3]

            # Parse the JSON
            content = json.loads(text)

            # Ensure verse is properly formatted
            daily_verse = format_verse(content.get('daily_verse', ''))

            # Check verse uniqueness
            reference = daily_verse.split('(')[-1].split(')')[0].strip()
            if is_verse_unique(daily_verse, st.session_state.previous_verses):
                # Update previous verses, keeping only the last 10
                st.session_state.previous_verses.append(reference)
                st.session_state.previous_verses = st.session_state.previous_verses[-10:]

                content['daily_verse'] = daily_verse
                return content

        except Exception as e:
            st.error(f"Attempt {attempt + 1} failed: {e}")

    # Fallback content if unique verse cannot be generated
    return {
        "daily_verse": "For God so loved the world that he gave his one and only Son, that whoever believes in him shall not perish but have eternal life. (John 3:16 NIV)",
        "daily_devotional": "Reflect on God's unconditional love and the hope it brings to our lives.",
        "prayer_guide": "Heavenly Father, guide my steps and fill my heart with your peace today.",
        "religious_insight": "The Christian faith is built on the transformative power of love, grace, and redemption."
    }


def format_verse(verse):
    """
    Format verse to ensure proper reference

    Args:
        verse (str): Input verse

    Returns:
        str: Formatted verse with reference
    """
    if not verse:
        return "No verse available."

    # If verse is already in the correct format, return it
    if "(" in verse and ")" in verse:
        return verse

    # If not, try to add a default reference
    return f"{verse} (Ephesians 2:8-9 NIV)"

# Load or initialize preferences
def load_preferences():
    if 'preferences' not in st.session_state:
        st.session_state['preferences'] = {
            "bible_version": "New International Version (NIV)",
            "themes": []
        }
    return st.session_state['preferences']


def save_preferences(preferences):
    st.session_state["preferences"] = preferences


# Streamlit App
def main():
    st.title("Daily Grace")
    st.write(
        "Discover inspiration, comfort, and guidance every day with Daily Grace. Begin each day with grace and grow in your walk with God.")

    # Load preferences
    preferences = load_preferences()

    # Generate initial content based on preferences
    try:
        content = get_spiritual_content(
            bible_version=preferences.get("bible_version"),
            themes=preferences.get("themes")
        )

        # Display Content Sections
        st.subheader("📖 Daily Bible Verse")
        st.write(content.get("daily_verse", "No verse available today."))

        st.subheader("💭 Daily Devotional")
        st.write(content.get("daily_devotional", "Today's devotional could not be generated."))

        st.subheader("🙏 Prayer Guide")
        st.write(content.get("prayer_guide", "A simple prayer for guidance."))

        st.subheader("🕊️ Religious Insight")
        st.write(content.get("religious_insight", "Each day is a gift from God."))

    except Exception as e:
        st.error(f"An error occurred: {e}")

    # Hyperlink to open preferences in the sidebar
    st.markdown(
        '<a href="#preferences" style="text-decoration:none;color:blue;">Customize Your Spiritual Journey (Open Sidebar) →</a>',
        unsafe_allow_html=True
    )

    # Sidebar for Preferences
    with st.sidebar:
        st.header("Preferences")
        st.markdown('<div id="preferences"></div>', unsafe_allow_html=True)  # Anchor for scrolling

        bible_version = st.selectbox(
            "Preferred Bible Version",
            [
                "New International Version (NIV)",
                "King James Version (KJV)",
                "English Standard Version (ESV)",
                "New Living Translation (NLT)",
                "New American Standard Bible (NASB)",
                "The Message (MSG)",
                "Other"
            ],
            index=0 if not preferences.get("bible_version") else ["New International Version (NIV)",
                                                                  "King James Version (KJV)",
                                                                  "English Standard Version (ESV)",
                                                                  "New Living Translation (NLT)",
                                                                  "New American Standard Bible (NASB)",
                                                                  "The Message (MSG)", "Other"].index(
                preferences["bible_version"])
        )
        if bible_version == "Other":
            bible_version = st.text_input("Please specify your preferred Bible version",
                                          value=preferences.get("bible_version") or "")

        themes = st.multiselect(
            "Select Your Spiritual Themes",
            [
                "Strength", "Gratitude", "Forgiveness", "Love", "Hope",
                "Peace", "Courage", "Wisdom", "Joy", "Patience",
                "Humility", "Compassion", "Faith", "Mindfulness",
                "Purpose", "Healing", "Unity", "Growth",
                "Generosity", "Resilience"
            ],
            default=preferences.get("themes") or []
        )

        if st.button("Save Preferences"):
            preferences = {"bible_version": bible_version, "themes": themes}
            save_preferences(preferences)
            st.success("Preferences saved! Refresh the app to see personalized content.")


# Run the app
if __name__ == "__main__":
    main()