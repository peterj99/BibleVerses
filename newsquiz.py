import streamlit as st
import google.generativeai as genai
from google.cloud import texttospeech
from google.oauth2 import service_account
import vertexai
import json
import random
import time
import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
import numpy as np
from collections import Counter
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import hashlib
import re
import warnings
from difflib import SequenceMatcher
import pytz
import requests
from bs4 import BeautifulSoup
import json
import time
import re
import logging
import numpy as np
from datetime import datetime, timedelta
from urllib.parse import urljoin
from dataclasses import dataclass, field
import hashlib
from collections import Counter
import warnings
from difflib import SequenceMatcher
import pytz
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

# Suppress warnings for cleaner UI
warnings.filterwarnings("ignore")

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="Prime Time Quiz Night", page_icon="📺", layout="centered"
)

# Custom TV Show CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Montserrat:wght@400;700&display=swap');

    /* TV Studio Theme */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: white;
    }

    h1, h2, h3 {
        font-family: 'Bebas Neue', sans-serif;
        letter-spacing: 2px;
        text-shadow: 0 0 10px rgba(0, 255, 255, 0.7);
    }

    p, div, button {
        font-family: 'Montserrat', sans-serif;
    }

    /* Game Show Card */
    .tv-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 2px solid rgba(255, 255, 255, 0.2);
        border-radius: 20px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 0 20px rgba(0,0,0,0.5);
    }

    /* Persona Badges */
    .badge-professor { background: #2c3e50; border: 1px solid #34495e; color: #ecf0f1; padding: 5px 10px; border-radius: 5px; font-weight: bold; }
    .badge-steve { background: #d35400; border: 1px solid #e67e22; color: white; padding: 5px 10px; border-radius: 5px; font-weight: bold; }
    .badge-regis { background: #2980b9; border: 1px solid #3498db; color: white; padding: 5px 10px; border-radius: 5px; font-weight: bold; }

    /* Answer Buttons (Custom Streamlit Button Override) */
    .stButton button {
        width: 100%;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.3);
        background: rgba(0,0,0,0.3);
        color: white;
        font-size: 18px;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: scale(1.02);
        background: rgba(255,255,255,0.1);
        border-color: #00ffff;
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.4);
    }

    /* Lifeline Buttons */
    .lifeline-btn {
        font-size: 12px;
        text-align: center;
        cursor: pointer;
    }
    
    /* Animations */
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(0, 255, 255, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(0, 255, 255, 0); }
        100% { box-shadow: 0 0 0 0 rgba(0, 255, 255, 0); }
    }
    .live-indicator {
        animation: pulse 2s infinite;
        color: red;
        font-weight: bold;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# --- 2. AUTHENTICATION SETUP ---
try:
    # 1. Gemini Setup
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)

    # 2. Google Cloud Setup (Service Account)
    GCP_CREDS = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"]
    )

    # 3. Vertex AI Setup (for potential Image Gen)
    vertexai.init(
        project=st.secrets["GCP_PROJECT_ID"],
        location=st.secrets["GCP_LOCATION"],
        credentials=GCP_CREDS,
    )

    # 4. TTS Client Setup (using Service Account)
    tts_client = texttospeech.TextToSpeechClient(credentials=GCP_CREDS)

    # 5. TTS Function (Helper)
    def generate_tts_audio(text: str, persona: str) -> Optional[bytes]:
        """Generates audio using Google Cloud TTS"""
        try:
            # Map persona to specific voice characteristics if available
            # For standard TTS, we'll use a neutral high-quality voice
            synthesis_input = texttospeech.SynthesisInput(text=text)

            # Select voice (en-US-Journey-D is a good conversational voice)
            voice = texttospeech.VoiceSelectionParams(
                language_code="en-US", name="en-US-Journey-D"
            )

            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16
            )

            response = tts_client.synthesize_speech(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )
            return response.audio_content
        except Exception as e:
            print(f"TTS Error: {e}")
            return None

except Exception as e:
    st.error(f"⚠️ Configuration Error: {e}")
    st.stop()


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class NewsArticle:
    """Complete news article with topic and headline"""

    id: str
    source: str
    topic: str
    headline: str
    link: Optional[str]
    position: int
    timestamp: str

    # Analysis fields
    embedding: Optional[np.ndarray] = None
    entities: Set[str] = field(default_factory=set)
    story_category: str = "general"
    cluster_id: Optional[int] = None

    def get_analysis_text(self) -> str:
        """Get text for analysis (topic + headline)"""
        if self.headline and self.headline != self.topic:
            return f"{self.topic}. {self.headline}"
        return self.topic

    def to_dict(self):
        data = {
            "id": self.id,
            "source": self.source,
            "topic": self.topic,
            "headline": self.headline,
            "link": self.link,
            "position": self.position,
            "timestamp": self.timestamp,
            "entities": list(self.entities),
            "story_category": self.story_category,
            "cluster_id": self.cluster_id,
            "analysis_text": self.get_analysis_text(),
        }
        if self.embedding is not None:
            data["embedding"] = self.embedding.tolist()
        return data


@dataclass
class StoryCluster:
    """Story cluster with essential information"""

    cluster_id: int
    articles: List[NewsArticle]
    representative_headline: str
    confidence_score: float
    sources: List[str]
    category: str
    trending_score: float


# ============================================================================
# NEWS SCRAPERS (Complete Implementation)
# ============================================================================


class CNNScraper:
    """CNN News Scraper - Topic + Headline only"""

    def __init__(self):
        self.source_name = "CNN"
        self.base_url = "https://edition.cnn.com"
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        )
        self.logger = logging.getLogger("CNN_Scraper")

    def scrape(self) -> List[NewsArticle]:
        """Scrape CNN ribbon topics and headlines"""
        articles = []

        try:
            self.logger.info("🔍 Starting CNN scraping...")
            response = self.session.get(self.base_url, timeout=20)

            if response.status_code != 200:
                self.logger.error(
                    f"❌ CNN homepage failed: status {response.status_code}"
                )
                return articles

            soup = BeautifulSoup(response.content, "html.parser")
            self.logger.info("✅ CNN homepage loaded successfully")

            # Extract ribbon topics
            ribbon_selector = (
                ".container_ribbon__cards-wrapper .container__headline-text"
            )
            ribbon_elements = soup.select(ribbon_selector)

            self.logger.info(f"🔍 Found {len(ribbon_elements)} CNN ribbon elements")

            for idx, element in enumerate(ribbon_elements[:8]):  # Limit to 8
                try:
                    topic = element.get_text(strip=True)
                    if not topic or len(topic) < 10:
                        self.logger.debug(f"⚠️ Skipping short CNN topic: '{topic}'")
                        continue

                    # Get link
                    link_element = element.find_parent("a")
                    link = None
                    if link_element:
                        href = link_element.get("href")
                        if href:
                            link = urljoin(self.base_url, href)

                    # For CNN, try to get headline from link if topic is short
                    headline = topic
                    if link and len(topic) < 30:
                        try:
                            self.logger.debug(
                                f"🔍 Fetching headline for short CNN topic: '{topic}'"
                            )
                            headline = self._fetch_headline(link) or topic
                            time.sleep(0.3)  # Brief delay
                        except Exception as e:
                            self.logger.debug(f"⚠️ Failed to fetch CNN headline: {e}")

                    article = NewsArticle(
                        id=hashlib.md5(
                            f"{topic}_CNN_{datetime.now().strftime('%Y-%m-%d')}".encode()
                        ).hexdigest()[:12],
                        source="CNN",
                        topic=topic,
                        headline=headline,
                        link=link,
                        position=idx + 1,
                        timestamp=datetime.now().isoformat(),
                    )

                    articles.append(article)
                    self.logger.info(f"✅ CNN {idx+1}: '{topic}' -> '{headline}'")

                except Exception as e:
                    self.logger.error(f"❌ Error processing CNN article {idx}: {e}")

            self.logger.info(f"✅ CNN scraping complete: {len(articles)} articles")

        except Exception as e:
            self.logger.error(f"❌ CNN scraping failed: {e}")

        return articles

    def _fetch_headline(self, url: str) -> Optional[str]:
        """Fetch headline from article URL"""
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code != 200:
                return None

            soup = BeautifulSoup(response.content, "html.parser")

            # Try multiple selectors
            for selector in ["h1", "h1.headline", ".headline h1", "title"]:
                element = soup.select_one(selector)
                if element:
                    headline = element.get_text(strip=True)
                    if headline and len(headline) > 15:
                        # Clean CNN branding
                        headline = (
                            headline.replace(" - CNN", "").replace(" | CNN", "").strip()
                        )
                        return headline

            return None

        except Exception:
            return None


class ABCScraper:
    """ABC News Scraper - Topic + Headline only"""

    def __init__(self):
        self.source_name = "ABC"
        self.base_url = "https://abcnews.go.com"
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        )
        self.logger = logging.getLogger("ABC_Scraper")

    def scrape(self) -> List[NewsArticle]:
        """Scrape ABC topics and headlines"""
        articles = []

        try:
            self.logger.info("🔍 Starting ABC scraping...")
            response = self.session.get(self.base_url, timeout=20)

            if response.status_code != 200:
                self.logger.error(
                    f"❌ ABC homepage failed: status {response.status_code}"
                )
                return articles

            soup = BeautifulSoup(response.content, "html.parser")
            self.logger.info("✅ ABC homepage loaded successfully")

            # Multiple selectors for ABC
            selectors = [
                "a.AnchorLink.EeYLf.klTtn.gqvTK.rXPFS.XANA.mmTGz.JQiqM.OXKHx.pBQn.eyDwZ",
                "a.zZygg.UbGlr.iFzkS.qdXbA.WCDhQ.DbOXS.tqUtK.GpWVU.iJYzE",
                "h2 a",
                "h3 a",
            ]

            seen_topics = set()

            for selector in selectors:
                elements = soup.select(selector)
                self.logger.debug(
                    f"🔍 ABC selector '{selector}' found {len(elements)} elements"
                )

                for element in elements:
                    if len(articles) >= 8:  # Limit to 8
                        break

                    try:
                        topic = element.get_text(strip=True)
                        if not topic or len(topic) < 10 or topic in seen_topics:
                            continue

                        seen_topics.add(topic)

                        # Get link
                        href = element.get("href")
                        link = urljoin(self.base_url, href) if href else None

                        # For ABC, use topic as headline unless we can get better
                        headline = topic
                        if link and len(topic) < 40:
                            try:
                                self.logger.debug(
                                    f"🔍 Fetching headline for ABC topic: '{topic[:30]}...'"
                                )
                                headline = self._fetch_headline(link) or topic
                                time.sleep(0.3)
                            except Exception as e:
                                self.logger.debug(
                                    f"⚠️ Failed to fetch ABC headline: {e}"
                                )

                        article = NewsArticle(
                            id=hashlib.md5(
                                f"{topic}_ABC_{datetime.now().strftime('%Y-%m-%d')}".encode()
                            ).hexdigest()[:12],
                            source="ABC",
                            topic=topic[:100] + "..." if len(topic) > 100 else topic,
                            headline=headline,
                            link=link,
                            position=len(articles) + 1,
                            timestamp=datetime.now().isoformat(),
                        )

                        articles.append(article)
                        self.logger.info(
                            f"✅ ABC {len(articles)}: '{topic[:30]}...' -> '{headline[:30]}...'"
                        )

                    except Exception as e:
                        self.logger.error(f"❌ Error processing ABC element: {e}")

                if len(articles) >= 8:
                    break

            self.logger.info(f"✅ ABC scraping complete: {len(articles)} articles")

        except Exception as e:
            self.logger.error(f"❌ ABC scraping failed: {e}")

        return articles

    def _fetch_headline(self, url: str) -> Optional[str]:
        """Fetch headline from article URL"""
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code != 200:
                return None

            soup = BeautifulSoup(response.content, "html.parser")

            for selector in ['h1[data-module="ArticleHeader"]', "h1", "title"]:
                element = soup.select_one(selector)
                if element:
                    headline = element.get_text(strip=True)
                    if headline and len(headline) > 15:
                        headline = re.sub(
                            r"\s*[-|]\s*ABC News.*$", "", headline, flags=re.I
                        ).strip()
                        return headline

            return None

        except Exception:
            return None


class FoxScraper:
    """Fox News Scraper - Topic + Headline extraction"""

    def __init__(self):
        self.source_name = "Fox"
        self.base_url = "https://www.foxnews.com"
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        )
        self.logger = logging.getLogger("Fox_Scraper")

    def scrape(self) -> List[NewsArticle]:
        """Scrape Fox News topics and extract headlines"""
        articles = []

        try:
            self.logger.info("🔍 Starting Fox News scraping...")
            response = self.session.get(self.base_url, timeout=20)

            if response.status_code != 200:
                self.logger.error(
                    f"❌ Fox homepage failed: status {response.status_code}"
                )
                return articles

            soup = BeautifulSoup(response.content, "html.parser")
            self.logger.info("✅ Fox homepage loaded successfully")

            # Find trending anchors using multiple strategies
            anchors = self._find_trending_anchors(soup)
            self.logger.info(f"🔍 Found {len(anchors)} Fox trending anchors")

            seen_topics = set()

            for idx, anchor in enumerate(anchors[:8]):  # Limit to 8
                try:
                    topic = anchor.get_text(strip=True)
                    if not topic or len(topic) < 5 or topic in seen_topics:
                        continue

                    seen_topics.add(topic)

                    href = anchor.get("href")
                    link = urljoin(self.base_url, href) if href else None

                    # Extract headline from topic page
                    headline = self._extract_headline_from_topic_page(link) or topic

                    # Rate limiting
                    time.sleep(0.5)

                    article = NewsArticle(
                        id=hashlib.md5(
                            f"{topic}_Fox_{datetime.now().strftime('%Y-%m-%d')}".encode()
                        ).hexdigest()[:12],
                        source="Fox",
                        topic=topic,
                        headline=headline,
                        link=link,
                        position=len(articles) + 1,
                        timestamp=datetime.now().isoformat(),
                    )

                    articles.append(article)
                    self.logger.info(
                        f"✅ Fox {len(articles)}: '{topic[:30]}...' -> '{headline[:40]}...'"
                    )

                except Exception as e:
                    self.logger.error(f"❌ Error processing Fox anchor {idx}: {e}")

            self.logger.info(f"✅ Fox scraping complete: {len(articles)} articles")

        except Exception as e:
            self.logger.error(f"❌ Fox scraping failed: {e}")

        return articles

    def _extract_headline_from_topic_page(self, topic_link: str) -> Optional[str]:
        """Extract headline from Fox topic page using multi-strategy approach"""
        if not topic_link:
            return None

        try:
            self.logger.debug(f"🔍 Fetching headline from Fox topic page: {topic_link}")
            response = self.session.get(topic_link, timeout=10)
            if response.status_code != 200:
                return None

            soup = BeautifulSoup(response.text, "html.parser")

            # Strategy 1: Look for articles with date patterns first (most recent)
            anchors = soup.find_all("a", href=True)
            date_pattern_anchors = []
            for a in anchors:
                if re.search(r"/\d{4}/\d{2}/\d{2}/", a["href"]):
                    date_pattern_anchors.append(a)

            if date_pattern_anchors:
                # Get headline from first date-pattern article
                first_article_href = date_pattern_anchors[0]["href"]
                article_link = urljoin(self.base_url, first_article_href)
                headline = self._fetch_article_headline(article_link)
                if headline:
                    return headline

            # Strategy 2: Try common article selectors
            list_selectors = [
                "article a[href]",
                "h2 a[href]",
                "h3 a[href]",
                ".article-list a[href]",
                ".collection a[href]",
            ]

            for sel in list_selectors:
                candidates = soup.select(sel)
                if candidates:
                    # Try first candidate
                    first_candidate_href = candidates[0].get("href")
                    if first_candidate_href:
                        article_link = urljoin(self.base_url, first_candidate_href)
                        headline = self._fetch_article_headline(article_link)
                        if headline:
                            return headline

            return None

        except Exception as e:
            self.logger.debug(f"⚠️ Error extracting Fox headline: {e}")
            return None

    def _fetch_article_headline(self, article_url: str) -> Optional[str]:
        """Fetch headline from Fox article page"""
        try:
            response = self.session.get(article_url, timeout=10)
            if response.status_code != 200:
                return None

            soup = BeautifulSoup(response.text, "html.parser")

            # Try multiple headline selectors
            headline_selectors = [
                "h1",
                "h1.headline",
                ".headline h1",
                ".article-title",
                "article h1",
                ".content h1",
                ".article-hero__title",
            ]

            for sel in headline_selectors:
                headline_tag = soup.select_one(sel)
                if headline_tag and headline_tag.get_text(strip=True):
                    headline = headline_tag.get_text(strip=True)
                    # Clean Fox News suffixes
                    headline = re.sub(
                        r"\s*[-|]\s*Fox News.*$", "", headline, flags=re.I
                    ).strip()
                    if len(headline) > 10:
                        return headline

            # Fallback to title tag
            title = soup.find("title")
            if title and title.get_text(strip=True):
                headline = title.get_text(strip=True)
                headline = re.sub(
                    r"\s*[-|]\s*Fox News.*$", "", headline, flags=re.I
                ).strip()
                return headline

            return None

        except Exception:
            return None

    def _find_trending_anchors(self, soup: BeautifulSoup) -> List:
        """Find trending anchors using multiple strategies"""
        anchors = []

        # Strategy 1: Known selectors
        selectors = [
            ".trending-item a",
            ".hot-topics a",
            "nav.hot-topics ul.trending-item li a",
            ".nav-item-label",
            ".trending a",
            ".module--trending a",
        ]

        for sel in selectors:
            found = soup.select(sel)
            if found:
                self.logger.debug(f"🔍 Fox selector '{sel}' found {len(found)} anchors")
                anchors.extend(found)

        if anchors:
            return self._deduplicate_anchors(anchors)

        # Strategy 2: Find "Trending" headings
        trending_headings = soup.find_all(
            lambda tag: tag.name in ("h2", "h3", "h4", "strong", "span")
            and "trending" in tag.get_text().lower()
        )

        for heading in trending_headings:
            parent = heading.find_parent()
            if parent:
                parent_anchors = parent.select("a[href]")
                if parent_anchors:
                    self.logger.debug(
                        f"🔍 Found {len(parent_anchors)} anchors under trending heading"
                    )
                    anchors.extend(parent_anchors)

        if anchors:
            return self._deduplicate_anchors(anchors)

        # Strategy 3: Fallback - nav/aside anchors
        for container in soup.select("nav, aside")[:3]:
            container_anchors = container.select("a[href]")[:10]
            for anchor in container_anchors:
                text = anchor.get_text(strip=True)
                if text and len(text) > 5:
                    anchors.append(anchor)
            if anchors:
                break

        self.logger.debug(f"🔍 Fox fallback strategy found {len(anchors)} anchors")
        return self._deduplicate_anchors(anchors)

    def _deduplicate_anchors(self, anchors: List) -> List:
        """Remove duplicate anchors"""
        seen = set()
        unique = []
        for anchor in anchors:
            text = anchor.get_text(strip=True)
            href = anchor.get("href", "")
            key = (text, href)
            if key not in seen and text:
                seen.add(key)
                unique.append(anchor)
        return unique


class NBCScraper:
    """NBC News Scraper - Topic + Headline extraction"""

    def __init__(self):
        self.source_name = "NBC"
        self.base_url = "https://www.nbcnews.com"
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        )
        self.logger = logging.getLogger("NBC_Scraper")

    def scrape(self) -> List[NewsArticle]:
        """Scrape NBC topics and extract headlines"""
        articles = []

        try:
            self.logger.info("🔍 Starting NBC scraping...")
            response = self.session.get(self.base_url, timeout=20)

            if response.status_code != 200:
                self.logger.error(
                    f"❌ NBC homepage failed: status {response.status_code}"
                )
                return articles

            soup = BeautifulSoup(response.content, "html.parser")
            self.logger.info("✅ NBC homepage loaded successfully")

            # Find trending ribbon
            ribbon = soup.find("nav", class_="styles_quickLinks__qK_Tl")
            if not ribbon:
                self.logger.warning("⚠️ NBC trending ribbon not found")
                return articles

            self.logger.info("✅ NBC trending ribbon found")

            trending_links = ribbon.find_all("a", href=True)
            self.logger.info(f"🔍 Found {len(trending_links)} NBC trending links")

            for idx, link in enumerate(trending_links[:8]):  # Limit to 8
                try:
                    # Skip tipline
                    if "styles_tipline__" in " ".join(link.get("class", [])):
                        continue

                    topic = link.get_text(strip=True)
                    if not topic or len(topic) < 5:
                        continue

                    href = link.get("href")
                    link_url = urljoin(self.base_url, href) if href else None

                    # Extract headline from topic page
                    headline = self._extract_headline_from_topic_page(link_url) or topic

                    # Rate limiting
                    time.sleep(0.5)

                    article = NewsArticle(
                        id=hashlib.md5(
                            f"{topic}_NBC_{datetime.now().strftime('%Y-%m-%d')}".encode()
                        ).hexdigest()[:12],
                        source="NBC",
                        topic=topic,
                        headline=headline,
                        link=link_url,
                        position=len(articles) + 1,
                        timestamp=datetime.now().isoformat(),
                    )

                    articles.append(article)
                    self.logger.info(
                        f"✅ NBC {len(articles)}: '{topic}' -> '{headline[:40]}...'"
                    )

                except Exception as e:
                    self.logger.error(f"❌ Error processing NBC link {idx}: {e}")

            self.logger.info(f"✅ NBC scraping complete: {len(articles)} articles")

        except Exception as e:
            self.logger.error(f"❌ NBC scraping failed: {e}")

        return articles

    def _extract_headline_from_topic_page(self, topic_link: str) -> Optional[str]:
        """Extract headline from NBC topic page"""
        if not topic_link:
            return None

        try:
            self.logger.debug(f"🔍 Fetching headline from NBC topic page: {topic_link}")
            response = self.session.get(topic_link, timeout=10)
            if response.status_code != 200:
                return None

            soup = BeautifulSoup(response.text, "html.parser")

            # Use the corrected NBC headline selector
            headline_tag = soup.find("h1", class_="article-hero-headline__htag")
            if headline_tag and headline_tag.get_text(strip=True):
                headline = headline_tag.get_text(strip=True)
                # Clean NBC branding
                headline = re.sub(
                    r"\s*[-|]\s*NBC News.*$", "", headline, flags=re.I
                ).strip()
                return headline

            # Fallback to title tag
            title = soup.find("title")
            if title and title.get_text(strip=True):
                headline = title.get_text(strip=True)
                headline = re.sub(
                    r"\s*[-|]\s*NBC News.*$", "", headline, flags=re.I
                ).strip()
                return headline

            return None

        except Exception as e:
            self.logger.debug(f"⚠️ Error extracting NBC headline: {e}")
            return None


class APScraper:
    """AP News Scraper - Topic + Headline extraction"""

    def __init__(self):
        self.source_name = "AP"
        self.base_url = "https://apnews.com"
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        )
        self.logger = logging.getLogger("AP_Scraper")

    def scrape(self) -> List[NewsArticle]:
        """Scrape AP News topics and extract headlines"""
        articles = []

        try:
            self.logger.info("🔍 Starting AP News scraping...")
            response = self.session.get(self.base_url, timeout=20)

            if response.status_code != 200:
                self.logger.error(
                    f"❌ AP homepage failed: status {response.status_code}"
                )
                return articles

            soup = BeautifulSoup(response.content, "html.parser")
            self.logger.info("✅ AP homepage loaded successfully")

            # Find trending items
            trending_items = soup.select("div.PageList-items-item")
            self.logger.info(f"🔍 Found {len(trending_items)} AP trending items")

            for idx, item in enumerate(trending_items[:8]):  # Limit to 8
                try:
                    link_tag = item.select_one("a.Link.AnClick-TrendingLink")
                    if not link_tag:
                        continue

                    topic = link_tag.get_text(strip=True)
                    if not topic or len(topic) < 10:
                        continue

                    href = link_tag.get("href")
                    link_url = None
                    if href and not href.startswith("http"):
                        link_url = self.base_url + href
                    elif href:
                        link_url = href

                    # Extract headline from topic page
                    headline = self._extract_headline_from_topic_page(link_url) or topic

                    # Rate limiting
                    time.sleep(0.5)

                    article = NewsArticle(
                        id=hashlib.md5(
                            f"{topic}_AP_{datetime.now().strftime('%Y-%m-%d')}".encode()
                        ).hexdigest()[:12],
                        source="AP",
                        topic=topic,
                        headline=headline,
                        link=link_url,
                        position=len(articles) + 1,
                        timestamp=datetime.now().isoformat(),
                    )

                    articles.append(article)
                    self.logger.info(
                        f"✅ AP {len(articles)}: '{topic[:30]}...' -> '{headline[:40]}...'"
                    )

                except Exception as e:
                    self.logger.error(f"❌ Error processing AP item {idx}: {e}")

            self.logger.info(f"✅ AP scraping complete: {len(articles)} articles")

        except Exception as e:
            self.logger.error(f"❌ AP scraping failed: {e}")

        return articles

    def _extract_headline_from_topic_page(self, topic_link: str) -> Optional[str]:
        """Extract headline from AP topic page"""
        if not topic_link:
            return None

        try:
            self.logger.debug(f"🔍 Fetching headline from AP topic page: {topic_link}")
            response = self.session.get(topic_link, timeout=10)
            if response.status_code != 200:
                return None

            soup = BeautifulSoup(response.text, "html.parser")

            # Look for main headline
            headline_tag = soup.select_one("h1")
            if headline_tag and headline_tag.get_text(strip=True):
                headline = headline_tag.get_text(strip=True)
                # Clean AP branding
                headline = re.sub(
                    r"\s*[-|]\s*AP News.*$", "", headline, flags=re.I
                ).strip()
                return headline

            # Fallback to title tag
            title = soup.find("title")
            if title and title.get_text(strip=True):
                headline = title.get_text(strip=True)
                headline = re.sub(
                    r"\s*[-|]\s*AP News.*$", "", headline, flags=re.I
                ).strip()
                return headline

            return None

        except Exception as e:
            self.logger.debug(f"⚠️ Error extracting AP headline: {e}")
            return None


# ============================================================================
# ANALYSIS COMPONENTS
# ============================================================================


class SimpleEntityExtractor:
    """Simple entity extraction for topic + headline analysis"""

    def __init__(self):
        self.political_keywords = {
            "trump": ["trump", "donald trump"],
            "biden": ["biden", "joe biden"],
            "putin": ["putin", "vladimir putin"],
            "zelensky": ["zelensky", "zelenskyy"],
            "harris": ["harris", "kamala harris"],
            "ukraine": ["ukraine", "ukrainian"],
            "russia": ["russia", "russian"],
            "china": ["china", "chinese"],
            "white_house": ["white house", "whitehouse"],
            "congress": ["congress", "senate", "house"],
        }

        self.logger = logging.getLogger("EntityExtractor")

    def extract_entities(self, text: str) -> Set[str]:
        """Extract entities from text"""
        entities = set()
        text_lower = text.lower()

        try:
            # Political entities
            for normalized, variants in self.political_keywords.items():
                for variant in variants:
                    if variant in text_lower:
                        entities.add(normalized)
                        break

            # Proper names (capitalized words)
            names = re.findall(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b", text)
            for name in names:
                if len(name) > 3:
                    entities.add(name.lower())

            # Years and numbers
            years = re.findall(r"\b20[0-9]{2}\b", text)
            entities.update(years)

            self.logger.debug(
                f"Extracted {len(entities)} entities: {list(entities)[:5]}"
            )

        except Exception as e:
            self.logger.error(f"❌ Entity extraction error: {e}")

        return entities


class StoryAnalyzer:
    """Analyze articles using topic + headline"""

    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.logger = logging.getLogger("StoryAnalyzer")

        try:
            self.logger.info(f"🧠 Loading Sentence-BERT model: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.logger.info("✅ Sentence-BERT model loaded successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to load model: {e}")
            raise

        self.entity_extractor = SimpleEntityExtractor()

        # Category patterns
        self.category_patterns = {
            "politics": [
                "election",
                "vote",
                "campaign",
                "congress",
                "senate",
                "president",
                "biden",
                "trump",
                "republican",
                "democrat",
                "political",
            ],
            "international": [
                "ukraine",
                "russia",
                "china",
                "war",
                "putin",
                "zelensky",
                "nato",
                "foreign",
                "diplomat",
            ],
            "weather": [
                "hurricane",
                "storm",
                "tornado",
                "flood",
                "weather",
                "tropical",
            ],
            "crime": ["shooting", "murder", "arrest", "police", "killed", "crime"],
            "sports": ["game", "team", "player", "win", "score", "nfl", "nba"],
            "business": ["company", "stock", "market", "economy", "revenue"],
            "health": ["health", "medical", "covid", "vaccine", "hospital"],
            "tech": ["tech", "ai", "computer", "internet", "software"],
        }

    def analyze_article(self, article: NewsArticle) -> NewsArticle:
        """Analyze article using topic + headline"""
        try:
            analysis_text = article.get_analysis_text()
            self.logger.debug(f"🔍 Analyzing: '{analysis_text[:50]}...'")

            # Generate embedding
            article.embedding = self.model.encode(
                analysis_text, convert_to_tensor=False
            )

            # Extract entities
            article.entities = self.entity_extractor.extract_entities(analysis_text)

            # Categorize
            article.story_category = self._categorize_story(analysis_text)

            self.logger.debug(
                f"✅ Analysis complete: {article.story_category}, {len(article.entities)} entities"
            )

        except Exception as e:
            self.logger.error(f"❌ Error analyzing article {article.id}: {e}")

        return article

    def _categorize_story(self, text: str) -> str:
        """Categorize story based on keywords"""
        text_lower = text.lower()
        scores = {}

        for category, keywords in self.category_patterns.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                scores[category] = score

        return max(scores, key=scores.get) if scores else "general"


class SimilarityCalculator:
    """Calculate similarity between articles"""

    def __init__(self):
        self.logger = logging.getLogger("SimilarityCalculator")

        # Category-specific thresholds
        self.thresholds = {
            "politics": 0.25,
            "international": 0.25,
            "weather": 0.40,
            "sports": 0.45,
            "crime": 0.35,
            "business": 0.35,
            "health": 0.35,
            "tech": 0.35,
            "general": 0.35,
        }

    def calculate_semantic_similarity(
        self, article1: NewsArticle, article2: NewsArticle
    ) -> float:
        """Calculate semantic similarity with safety bounds"""
        if article1.embedding is None or article2.embedding is None:
            self.logger.debug("⚠️ One or both embeddings are None, returning 0.0")
            return 0.0

        try:
            emb1 = article1.embedding.reshape(1, -1)
            emb2 = article2.embedding.reshape(1, -1)

            # Calculate cosine similarity
            similarity = float(cosine_similarity(emb1, emb2)[0, 0])

            # Safety bounds: clamp to [0,1] to handle floating point precision issues
            similarity = max(0.0, min(1.0, similarity))

            self.logger.debug(f"🧮 Semantic similarity: {similarity:.6f}")
            return similarity

        except Exception as e:
            self.logger.error(f"❌ Error calculating semantic similarity: {e}")
            return 0.0

    def calculate_entity_overlap(
        self, article1: NewsArticle, article2: NewsArticle
    ) -> float:
        """Calculate entity overlap with safety bounds"""
        entities1 = article1.entities
        entities2 = article2.entities

        if not entities1 and not entities2:
            self.logger.debug("⚠️ No entities in either article")
            return 0.0

        try:
            intersection = len(entities1.intersection(entities2))
            union = len(entities1.union(entities2))

            overlap = intersection / union if union > 0 else 0.0

            # Safety bounds: ensure result is in [0,1]
            overlap = max(0.0, min(1.0, overlap))

            self.logger.debug(
                f"🏷️ Entity overlap: {overlap:.6f} ({intersection}/{union})"
            )
            return overlap

        except Exception as e:
            self.logger.error(f"❌ Error calculating entity overlap: {e}")
            return 0.0

    def calculate_combined_similarity(
        self, trending_article: NewsArticle, cms_article_text: str
    ) -> float:
        """Calculate combined similarity between trending article and CMS article text"""
        try:
            # Create temporary article object for CMS text
            cms_article = NewsArticle(
                id="temp",
                source="cms",
                topic=cms_article_text,
                headline=cms_article_text,
                link=None,
                position=0,
                timestamp="",
            )

            # Analyze the CMS article
            analyzer = StoryAnalyzer()
            cms_article = analyzer.analyze_article(cms_article)

            # Calculate similarities
            semantic_sim = self.calculate_semantic_similarity(
                trending_article, cms_article
            )
            entity_overlap = self.calculate_entity_overlap(
                trending_article, cms_article
            )

            # Combined score (70% semantic, 30% entity)
            combined_score = semantic_sim * 0.7 + entity_overlap * 0.3

            return max(0.0, min(1.0, combined_score))
        except Exception as e:
            self.logger.error(f"❌ Error calculating combined similarity: {e}")
            return 0.0

    def are_same_story(
        self, article1: NewsArticle, article2: NewsArticle
    ) -> Tuple[bool, float]:
        """Determine if two articles are about the same story with robust error handling"""

        try:
            # Calculate similarities with safety bounds
            semantic_sim = self.calculate_semantic_similarity(article1, article2)
            entity_overlap = self.calculate_entity_overlap(article1, article2)

            # Validate similarity values
            if not (0.0 <= semantic_sim <= 1.0):
                self.logger.warning(
                    f"⚠️ Invalid semantic similarity: {semantic_sim}, clamping to [0,1]"
                )
                semantic_sim = max(0.0, min(1.0, semantic_sim))

            if not (0.0 <= entity_overlap <= 1.0):
                self.logger.warning(
                    f"⚠️ Invalid entity overlap: {entity_overlap}, clamping to [0,1]"
                )
                entity_overlap = max(0.0, min(1.0, entity_overlap))

            # Check for political entities
            political_entities = {
                "trump",
                "biden",
                "putin",
                "zelensky",
                "ukraine",
                "russia",
            }
            political_ents1 = article1.entities.intersection(political_entities)
            political_ents2 = article2.entities.intersection(political_entities)
            has_political_overlap = (
                len(political_ents1.intersection(political_ents2)) > 0
            )

            # Base weighted combination (ensure stays <= 1.0)
            base_score = semantic_sim * 0.7 + entity_overlap * 0.3

            # Add small bonuses (keep total under 1.0)
            cross_source_bonus = 0.03 if article1.source != article2.source else 0.0
            category_bonus = (
                0.02
                if (
                    article1.story_category == article2.story_category
                    and article1.story_category != "general"
                )
                else 0.0
            )

            # Calculate combined score with safety bounds
            combined_score = base_score + cross_source_bonus + category_bonus

            # CRITICAL: Ensure combined score never exceeds 1.0
            combined_score = max(0.0, min(1.0, combined_score))

            # Get threshold
            category = (
                article1.story_category
                if article1.story_category != "general"
                else article2.story_category
            )
            threshold = self.thresholds.get(category, 0.35)

            # Lower threshold for political stories with shared entities
            if has_political_overlap and category in ["politics", "international"]:
                threshold = 0.20
                self.logger.debug(
                    f"🏛️ Political story detected, lowered threshold to {threshold}"
                )

            # Final decision
            is_same = combined_score >= threshold

            # Detailed logging for debugging
            self.logger.debug(
                f"📊 Similarity breakdown: semantic={semantic_sim:.3f}, "
                f"entity={entity_overlap:.3f}, base={base_score:.3f}, "
                f"bonuses={cross_source_bonus + category_bonus:.3f}, "
                f"final={combined_score:.3f}, threshold={threshold:.3f}"
            )

            if is_same:
                self.logger.debug(
                    f"✅ MATCH: {article1.topic[:30]}... <-> {article2.topic[:30]}... "
                    f"(score={combined_score:.3f}, threshold={threshold:.3f})"
                )

            return is_same, combined_score

        except Exception as e:
            self.logger.error(f"❌ Error in story comparison: {e}")
            return False, 0.0


class ClusteringEngine:
    """Cluster articles using DBSCAN"""

    def __init__(self):
        self.logger = logging.getLogger("ClusteringEngine")
        self.similarity_calculator = SimilarityCalculator()

    def cluster_articles(self, articles: List[NewsArticle]) -> List[StoryCluster]:
        """Cluster articles using DBSCAN with robust error handling"""
        self.logger.info(f"🎯 Starting clustering for {len(articles)} articles")

        if len(articles) < 2:
            self.logger.info("📝 Too few articles for clustering, creating singletons")
            return self._create_singleton_clusters(articles)

        # Filter valid articles
        valid_articles = [a for a in articles if a.embedding is not None]
        invalid_count = len(articles) - len(valid_articles)

        if invalid_count > 0:
            self.logger.warning(
                f"⚠️ Skipping {invalid_count} articles with invalid embeddings"
            )

        self.logger.info(
            f"🎯 Clustering {len(valid_articles)} articles with valid embeddings"
        )

        if len(valid_articles) < 2:
            self.logger.info(
                "📝 Too few valid articles for clustering, creating singletons"
            )
            return self._create_singleton_clusters(articles)

        try:
            # Create distance matrix with robust error handling
            distance_matrix = self._create_distance_matrix(valid_articles)

            # Validate distance matrix before DBSCAN
            if distance_matrix.shape[0] != len(valid_articles):
                raise ValueError(
                    f"Distance matrix shape {distance_matrix.shape} doesn't match article count {len(valid_articles)}"
                )

            # DBSCAN parameters
            eps = 0.65  # Distance threshold
            min_samples = 2

            if len(valid_articles) < 10:
                eps = 0.70  # More lenient for small datasets
                self.logger.info("📏 Adjusted eps for small dataset")

            self.logger.info(
                f"🎯 Running DBSCAN with eps={eps:.3f}, min_samples={min_samples}"
            )

            # Run DBSCAN with error handling
            clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
            cluster_labels = clustering.fit_predict(distance_matrix)

            # Analyze results
            unique_labels = set(cluster_labels)
            n_clusters = len(unique_labels) - (1 if -1 in cluster_labels else 0)
            n_noise = list(cluster_labels).count(-1)

            self.logger.info(
                f"🎯 DBSCAN result: {n_clusters} clusters, {n_noise} noise points"
            )

            # Create clusters
            clusters = []

            # Process actual clusters
            for cluster_id in unique_labels:
                if cluster_id == -1:
                    continue

                cluster_mask = cluster_labels == cluster_id
                cluster_articles = [
                    valid_articles[i]
                    for i in range(len(valid_articles))
                    if cluster_mask[i]
                ]

                if not cluster_articles:
                    self.logger.warning(f"⚠️ Empty cluster {cluster_id}, skipping")
                    continue

                for article in cluster_articles:
                    article.cluster_id = cluster_id

                try:
                    cluster = self._analyze_cluster(cluster_id, cluster_articles)
                    clusters.append(cluster)
                except Exception as e:
                    self.logger.error(f"❌ Error analyzing cluster {cluster_id}: {e}")

            # Handle noise as singletons
            noise_indices = [i for i, label in enumerate(cluster_labels) if label == -1]
            for i, noise_idx in enumerate(noise_indices):
                try:
                    article = valid_articles[noise_idx]
                    article.cluster_id = -(i + 1)

                    singleton = StoryCluster(
                        cluster_id=-(i + 1),
                        articles=[article],
                        representative_headline=article.headline,
                        confidence_score=1.0,
                        sources=[article.source],
                        category=article.story_category,
                        trending_score=0.0,
                    )
                    clusters.append(singleton)
                except Exception as e:
                    self.logger.error(f"❌ Error creating singleton cluster: {e}")

            # Handle articles that weren't processed due to invalid embeddings
            processed_ids = {a.id for a in valid_articles}
            unprocessed_articles = [a for a in articles if a.id not in processed_ids]

            for i, article in enumerate(unprocessed_articles):
                article.cluster_id = -(len(noise_indices) + i + 1)
                try:
                    singleton = StoryCluster(
                        cluster_id=article.cluster_id,
                        articles=[article],
                        representative_headline=article.headline or article.topic,
                        confidence_score=1.0,
                        sources=[article.source],
                        category=article.story_category,
                        trending_score=0.0,
                    )
                    clusters.append(singleton)
                except Exception as e:
                    self.logger.error(
                        f"❌ Error creating singleton for unprocessed article: {e}"
                    )

            self.logger.info(f"✅ Clustering complete: {len(clusters)} total clusters")
            return clusters

        except Exception as e:
            self.logger.error(f"❌ Clustering failed: {e}")
            self.logger.info("📝 Falling back to singleton clusters")
            return self._create_singleton_clusters(articles)

    def _create_distance_matrix(self, articles: List[NewsArticle]) -> np.ndarray:
        """Create distance matrix for DBSCAN with robust error handling"""
        n_articles = len(articles)
        distance_matrix = np.ones((n_articles, n_articles), dtype=np.float64)

        self.logger.info(f"🧮 Computing distance matrix for {n_articles} articles...")

        similar_pairs = 0
        total_comparisons = 0
        invalid_similarities = 0

        for i in range(n_articles):
            distance_matrix[i, i] = 0.0  # Self-distance is always 0

            for j in range(i + 1, n_articles):
                total_comparisons += 1

                try:
                    is_same, similarity = self.similarity_calculator.are_same_story(
                        articles[i], articles[j]
                    )

                    # Validate similarity value
                    if not (0.0 <= similarity <= 1.0):
                        self.logger.warning(
                            f"⚠️ Invalid similarity {similarity} for articles {i}-{j}, clamping"
                        )
                        similarity = max(0.0, min(1.0, similarity))
                        invalid_similarities += 1

                    # Convert similarity to distance: distance = 1 - similarity
                    distance = 1.0 - similarity

                    # Ensure distance is valid for DBSCAN
                    distance = max(0.0, min(1.0, distance))

                    # Set symmetric matrix values
                    distance_matrix[i, j] = distance
                    distance_matrix[j, i] = distance

                    if is_same:
                        similar_pairs += 1

                except Exception as e:
                    self.logger.error(
                        f"❌ Error computing similarity for articles {i}-{j}: {e}"
                    )
                    # Set default distance for failed comparisons
                    distance_matrix[i, j] = 1.0
                    distance_matrix[j, i] = 1.0

        # Final validation: ensure all values are non-negative and finite
        if np.any(distance_matrix < 0):
            self.logger.warning("⚠️ Found negative distances, clipping to 0")
            distance_matrix = np.clip(distance_matrix, 0.0, 1.0)

        if not np.all(np.isfinite(distance_matrix)):
            self.logger.warning("⚠️ Found non-finite distances, replacing with 1.0")
            distance_matrix = np.nan_to_num(
                distance_matrix, nan=1.0, posinf=1.0, neginf=1.0
            )

        # Log statistics
        self.logger.info(
            f"🧮 Distance matrix complete: {similar_pairs}/{total_comparisons} similar pairs"
        )
        if invalid_similarities > 0:
            self.logger.warning(
                f"⚠️ Fixed {invalid_similarities} invalid similarity values"
            )

        return distance_matrix

    def _analyze_cluster(
        self, cluster_id: int, articles: List[NewsArticle]
    ) -> StoryCluster:
        """Analyze individual cluster"""

        # Choose representative (longest headline)
        representative = max(articles, key=lambda x: len(x.headline))

        # Calculate confidence
        if len(articles) > 1:
            similarities = []
            for i in range(len(articles)):
                for j in range(i + 1, len(articles)):
                    _, sim = self.similarity_calculator.are_same_story(
                        articles[i], articles[j]
                    )
                    similarities.append(sim)
            confidence = np.mean(similarities) if similarities else 0.0
        else:
            confidence = 1.0

        # Get sources and category
        sources = list(set(article.source for article in articles))
        categories = [article.story_category for article in articles]
        dominant_category = (
            Counter(categories).most_common(1)[0][0] if categories else "general"
        )

        return StoryCluster(
            cluster_id=cluster_id,
            articles=articles,
            representative_headline=representative.headline,
            confidence_score=confidence,
            sources=sources,
            category=dominant_category,
            trending_score=0.0,
        )

    def _create_singleton_clusters(
        self, articles: List[NewsArticle]
    ) -> List[StoryCluster]:
        """Create singleton clusters"""
        clusters = []
        for i, article in enumerate(articles):
            article.cluster_id = -(i + 1)
            cluster = StoryCluster(
                cluster_id=-(i + 1),
                articles=[article],
                representative_headline=article.headline,
                confidence_score=1.0,
                sources=[article.source],
                category=article.story_category,
                trending_score=0.0,
            )
            clusters.append(cluster)
        return clusters


# ============================================================================
# UNIFIED NEWS CLUSTERING SYSTEM
# ============================================================================


class UnifiedNewsClusteringSystem:
    """Main system orchestrating all 5 news sources"""

    def __init__(self):
        self.logger = logging.getLogger("UnifiedSystem")
        self.logger.info("🚀 Initializing Unified News Clustering System")

        # Initialize all scrapers
        self.scrapers = {
            "CNN": CNNScraper(),
            "ABC": ABCScraper(),
            "Fox": FoxScraper(),
            "NBC": NBCScraper(),
            "AP": APScraper(),
        }

        # Initialize analysis components
        self.analyzer = StoryAnalyzer()
        self.clustering_engine = ClusteringEngine()

        self.logger.info("✅ All system components initialized")
        self.logger.info(f"📰 News sources: {list(self.scrapers.keys())}")

    def scrape_all_sources(self) -> List[NewsArticle]:
        """Scrape all 5 news sources"""
        all_articles = []
        scraping_summary = {}

        self.logger.info("📰 Starting multi-source scraping...")

        for source_name, scraper in self.scrapers.items():
            start_time = time.time()
            try:
                self.logger.info(f"📡 Scraping {source_name}...")
                articles = scraper.scrape()
                scraping_time = time.time() - start_time

                all_articles.extend(articles)
                scraping_summary[source_name] = {
                    "articles_count": len(articles),
                    "scraping_time": scraping_time,
                    "success": True,
                }

                self.logger.info(
                    f"✅ {source_name}: {len(articles)} articles in {scraping_time:.1f}s"
                )

                # Brief delay between sources
                time.sleep(0.5)

            except Exception as e:
                scraping_time = time.time() - start_time
                scraping_summary[source_name] = {
                    "articles_count": 0,
                    "scraping_time": scraping_time,
                    "success": False,
                    "error": str(e),
                }
                self.logger.error(f"❌ {source_name} scraping failed: {e}")

        # Summary
        total_articles = len(all_articles)
        successful_sources = sum(1 for s in scraping_summary.values() if s["success"])
        total_sources = len(self.scrapers)

        self.logger.info(
            f"📊 Scraping complete: {total_articles} articles from {successful_sources}/{total_sources} sources"
        )

        return all_articles

    def analyze_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Analyze all articles with robust error handling"""
        self.logger.info(f"🧠 Starting analysis for {len(articles)} articles...")

        analyzed = []
        stats = {
            "successful_embeddings": 0,
            "failed_embeddings": 0,
            "entity_extraction_success": 0,
            "category_distribution": Counter(),
        }

        for i, article in enumerate(articles):
            try:
                self.logger.debug(
                    f"🔍 Analyzing article {i+1}/{len(articles)}: {article.source} - {article.topic[:30]}..."
                )

                analyzed_article = self.analyzer.analyze_article(article)
                analyzed.append(analyzed_article)

                # Update stats
                if analyzed_article.embedding is not None:
                    stats["successful_embeddings"] += 1
                    # Validate embedding
                    if not np.all(np.isfinite(analyzed_article.embedding)):
                        self.logger.warning(
                            f"⚠️ Article {article.id} has non-finite embedding values"
                        )
                        analyzed_article.embedding = None
                        stats["successful_embeddings"] -= 1
                        stats["failed_embeddings"] += 1
                else:
                    stats["failed_embeddings"] += 1

                if analyzed_article.entities:
                    stats["entity_extraction_success"] += 1

                stats["category_distribution"][analyzed_article.story_category] += 1

            except Exception as e:
                self.logger.error(f"❌ Error analyzing article {article.id}: {e}")
                stats["failed_embeddings"] += 1
                analyzed.append(article)  # Add original article without analysis

        # Log stats
        success_rate = (
            stats["successful_embeddings"] / len(articles) * 100 if articles else 0
        )
        self.logger.info(
            f"🧠 Analysis complete: {stats['successful_embeddings']}/{len(articles)} "
            f"successful embeddings ({success_rate:.1f}%)"
        )

        return analyzed

    def cluster_stories(self, articles: List[NewsArticle]) -> List[StoryCluster]:
        """Cluster articles into stories"""
        clusters = self.clustering_engine.cluster_articles(articles)

        # Calculate trending scores
        for cluster in clusters:
            cluster.trending_score = self._calculate_trending_score(cluster)

        # Sort by trending score
        clusters.sort(key=lambda x: x.trending_score, reverse=True)

        multi_source = sum(1 for c in clusters if len(c.sources) > 1)
        self.logger.info(
            f"🎯 Clustering summary: {len(clusters)} clusters, {multi_source} multi-source stories"
        )

        return clusters

    def _calculate_trending_score(self, cluster: StoryCluster) -> float:
        """Calculate trending score"""
        source_count = len(cluster.sources)
        article_count = len(cluster.articles)

        # Position scoring
        position_scores = [
            max(0, 10 - article.position) for article in cluster.articles
        ]
        avg_position_score = np.mean(position_scores) if position_scores else 0

        # Bonuses
        multi_source_bonus = 5 if source_count > 1 else 0
        category_bonus = (
            3
            if cluster.category in ["politics", "international", "weather", "crime"]
            else 0
        )
        confidence_bonus = cluster.confidence_score * 2

        total_score = (
            source_count * 3
            + article_count * 1
            + avg_position_score
            + multi_source_bonus
            + category_bonus
            + confidence_bonus
        )

        return total_score

    def get_top_trending_stories(self, count: int = 5) -> List[StoryCluster]:
        """Get top N trending stories from all sources"""
        try:
            # Step 1: Scrape all sources
            articles = self.scrape_all_sources()
            if not articles:
                self.logger.error("❌ No articles scraped from any source")
                return []

            # Step 2: Analyze articles
            analyzed_articles = self.analyze_articles(articles)

            # Step 3: Cluster stories
            story_clusters = self.cluster_stories(analyzed_articles)

            if not story_clusters:
                self.logger.error("❌ No clusters created")
                return []

            # Get top stories
            top_clusters = story_clusters[:count]

            self.logger.info(
                f"\n🏆 TOP {len(top_clusters)} TRENDING STORIES IDENTIFIED:"
            )
            for i, cluster in enumerate(top_clusters, 1):
                self.logger.info(f"   {i}. {cluster.representative_headline}")
                self.logger.info(
                    f"      Score: {cluster.trending_score:.1f} | Sources: {', '.join(cluster.sources)} | Category: {cluster.category}"
                )

            return top_clusters

        except Exception as e:
            self.logger.error(f"❌ Error getting top trending stories: {e}")
            return []


# ============================================================================
# CONTENTFUL INTEGRATION CLASSES
# ============================================================================


class ContentfulClient:
    """Contentful API client for Top News management"""

    def __init__(self):
        self.space_id = SPACE_ID
        self.token = PERSONAL_ACCESS_TOKEN
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        self.logger = logging.getLogger("ContentfulClient")

    def get_recent_articles(self, hours_back: int = 6) -> List[Dict]:
        """Get ALL articles from last N hours (timezone aware for IST) with smart pagination"""
        self.logger.info(
            f"🔍 Fetching ALL Contentful articles from last {hours_back} hours..."
        )

        # Calculate IST time threshold
        ist_tz = pytz.timezone("Asia/Kolkata")
        utc_now = datetime.now(pytz.UTC)
        ist_now = utc_now.astimezone(ist_tz)
        threshold_ist = ist_now - timedelta(hours=hours_back)
        threshold_str = threshold_ist.strftime("%Y-%m-%dT%H:%M:%S+05:30")

        self.logger.info(f"   📅 Looking for articles published after: {threshold_str}")

        # Fetch ALL articles with smart pagination
        all_articles = []
        skip = 0
        limit = 100
        total_requests = 0
        max_requests = 20

        while total_requests < max_requests:
            total_requests += 1
            params = {
                "content_type": "swishNewsContentful",
                "fields.publishedDate[gte]": threshold_str,
                "order": "-fields.publishedDate",
                "limit": limit,
                "skip": skip,
            }

            try:
                self.logger.info(
                    f"   📡 Request {total_requests}: fetching {limit} articles (skip: {skip})"
                )
                response = requests.get(
                    f"https://api.contentful.com/spaces/{self.space_id}/entries",
                    headers=self.headers,
                    params=params,
                )

                if response.status_code == 200:
                    data = response.json()
                    batch_articles = data["items"]

                    if not batch_articles:
                        self.logger.info(
                            f"      🏁 No more articles found (empty batch)"
                        )
                        break

                    all_articles.extend(batch_articles)
                    self.logger.info(f"      ✅ Fetched {len(batch_articles)} articles")

                    if len(batch_articles) < limit:
                        self.logger.info(
                            f"      🏁 Reached end (got {len(batch_articles)} < {limit})"
                        )
                        break

                    skip += limit
                    time.sleep(0.2)

                else:
                    self.logger.error(f"❌ API call failed: {response.status_code}")
                    break

            except Exception as e:
                self.logger.error(
                    f"❌ Error fetching articles batch {total_requests}: {e}"
                )
                break

        self.logger.info(f"✅ Total articles retrieved: {len(all_articles)}")
        return all_articles

    def find_matching_article(
        self, scraped_headline: str, contentful_articles: List[Dict]
    ) -> Optional[Dict]:
        """Find best matching Contentful article and auto-assign General category if missing"""
        self.logger.info(f"🔍 Finding match for scraped headline:")
        self.logger.info(f"   🎯 Target: '{scraped_headline}'")

        if not contentful_articles:
            self.logger.error("❌ No Contentful articles to match against")
            return None

        thresholds = [0.50, 0.30]

        for threshold in thresholds:
            self.logger.info(
                f"   🎯 Trying {threshold*100:.0f}% similarity threshold..."
            )

            best_match = None
            best_score = 0
            candidates_checked = 0

            for article in contentful_articles:
                try:
                    headline_field = article["fields"].get("headline", {})
                    if isinstance(headline_field, dict):
                        contentful_headline = headline_field.get("en-US", "")
                    else:
                        contentful_headline = headline_field or ""

                    if not contentful_headline:
                        continue

                    candidates_checked += 1

                    clean_scraped = self._clean_headline(scraped_headline)
                    clean_contentful = self._clean_headline(contentful_headline)

                    similarity = SequenceMatcher(
                        None, clean_scraped, clean_contentful
                    ).ratio()

                    if similarity > best_score and similarity >= threshold:
                        best_score = similarity
                        best_match = article
                        self.logger.debug(
                            f"      💡 New best match ({similarity*100:.1f}%): {contentful_headline[:40]}..."
                        )

                except Exception as e:
                    self.logger.warning(f"   ⚠️ Error matching article: {e}")
                    continue

            self.logger.info(
                f"      📊 Checked {candidates_checked} articles at {threshold*100:.0f}% threshold"
            )

            if best_match:
                contentful_headline = best_match["fields"]["headline"]["en-US"]
                self.logger.info(f"✅ MATCH FOUND at {threshold*100:.0f}% threshold!")
                self.logger.info(f"   📊 Similarity: {best_score*100:.1f}%")
                self.logger.info(f"   🆔 Article ID: {best_match['sys']['id']}")

                self._ensure_article_has_categories(best_match["sys"]["id"])

                return best_match
            else:
                self.logger.info(
                    f"      ❌ No matches found at {threshold*100:.0f}% threshold"
                )

        self.logger.error("❌ No suitable match found (even at 30% threshold)")
        return None

    def _clean_headline(self, headline: str) -> str:
        """Clean headline for better matching"""
        clean = headline.lower()
        clean = " ".join(clean.split())
        clean = re.sub(r"[^\w\s]", " ", clean)
        clean = " ".join(clean.split())
        return clean

    def get_current_top_news(self) -> Optional[Dict]:
        """Get current Top News entry"""
        self.logger.info("🔍 Getting current Top News entry...")

        try:
            response = requests.get(
                f"https://api.contentful.com/spaces/{self.space_id}/entries/{TOP_NEWS_ENTRY_ID}",
                headers=self.headers,
            )

            if response.status_code == 200:
                entry = response.json()
                current_articles = entry["fields"]["news"]["en-US"]
                self.logger.info(f"✅ Current Top News entry retrieved")
                self.logger.info(f"   📋 Version: {entry['sys']['version']}")
                self.logger.info(
                    f"   🔗 Currently linked articles: {len(current_articles)}"
                )

                self.logger.info("   📄 Current Top News articles (positions 1-5):")
                for i, article_ref in enumerate(current_articles[:5], 1):
                    self.logger.info(f"      Position {i}: {article_ref['sys']['id']}")

                return entry
            else:
                self.logger.error(f"❌ Failed to get Top News: {response.status_code}")
                self.logger.error(f"   Response: {response.text}")
                return None

        except Exception as e:
            self.logger.error(f"❌ Error getting Top News: {e}")
            return None

    def update_top_news_positions_batch(self, position_updates: Dict[int, str]) -> bool:
        """Update multiple Top News positions with new articles, avoiding duplicates"""
        self.logger.info(
            f"🔄 Updating Top News positions: {list(position_updates.keys())}"
        )

        try:
            current_entry = self.get_current_top_news()
            if not current_entry:
                return False

            current_articles = current_entry["fields"]["news"]["en-US"]
            current_version = current_entry["sys"]["version"]

            existing_ids = [article["sys"]["id"] for article in current_articles]

            filtered_updates = {}
            for position, article_id in position_updates.items():
                if article_id in existing_ids:
                    existing_position = existing_ids.index(article_id) + 1
                    self.logger.warning(
                        f"⚠️ Article {article_id} already exists at position {existing_position}, skipping position {position}"
                    )
                else:
                    filtered_updates[position] = article_id

            if not filtered_updates:
                self.logger.warning("⚠️ No valid updates after duplicate filtering")
                return False

            self.logger.info(
                f"📝 Proceeding with updates for positions: {list(filtered_updates.keys())}"
            )

            updated_articles = current_articles.copy()

            max_position = max(filtered_updates.keys())
            while len(updated_articles) < max_position:
                updated_articles.append(None)

            for position, article_id in filtered_updates.items():
                array_index = position - 1

                old_article_id = "None"
                if array_index < len(current_articles):
                    old_article_id = current_articles[array_index]["sys"]["id"]

                new_article_ref = {
                    "sys": {"type": "Link", "linkType": "Entry", "id": article_id}
                }

                updated_articles[array_index] = new_article_ref
                self.logger.info(
                    f"   📍 Position {position}: {old_article_id} → {article_id}"
                )

            updated_articles = [a for a in updated_articles if a is not None]

            current_fields = current_entry["fields"].copy()
            current_fields["news"] = {"en-US": updated_articles}

            update_payload = {"fields": current_fields}

            update_headers = self.headers.copy()
            update_headers["X-Contentful-Version"] = str(current_version)

            self.logger.info(f"   🔄 Sending batch update to Contentful...")
            self.logger.info(f"      Using version: {current_version}")

            response = requests.put(
                f"https://api.contentful.com/spaces/{self.space_id}/entries/{TOP_NEWS_ENTRY_ID}",
                headers=update_headers,
                json=update_payload,
            )

            if response.status_code == 200:
                updated_entry = response.json()
                new_version = updated_entry["sys"]["version"]
                self.logger.info(f"✅ TOP NEWS BATCH UPDATE SUCCESSFUL!")
                self.logger.info(f"   📋 Version: {current_version} → {new_version}")
                self.logger.info(
                    f"   📍 Updated positions: {list(filtered_updates.keys())}"
                )
                self.logger.info(
                    f"   🔗 Total articles in Top News: {len(updated_articles)}"
                )

                self.logger.info(f"\n📝 Step 2: Publishing Top News container...")
                container_publish_success = self.publish_top_news_container()

                if container_publish_success:
                    self.logger.info(f"✅ Top News container published successfully!")
                else:
                    self.logger.warning(
                        f"⚠️ Top News updated but container publishing failed"
                    )

                return True
            else:
                self.logger.error(f"❌ Batch update failed: {response.status_code}")
                self.logger.error(f"   Response: {response.text}")
                return False

        except Exception as e:
            self.logger.error(f"❌ Error in batch update: {e}")
            return False

    def publish_top_news_container(self) -> bool:
        """Publish the Top News container entry itself"""
        self.logger.info(f"📝 Publishing Top News container: {TOP_NEWS_ENTRY_ID}")

        try:
            current_entry = self.get_current_top_news()
            if not current_entry:
                return False

            current_version = current_entry["sys"]["version"]

            if "publishedVersion" in current_entry["sys"]:
                published_version = current_entry["sys"]["publishedVersion"]
                if published_version == current_version:
                    self.logger.info(
                        f"✅ Top News container already published (v{current_version})"
                    )
                    return True

            url = f"https://api.contentful.com/spaces/{self.space_id}/entries/{TOP_NEWS_ENTRY_ID}/published"
            publish_headers = self.headers.copy()
            publish_headers["X-Contentful-Version"] = str(current_version)

            self.logger.info(
                f"   🚀 Publishing Top News container v{current_version}..."
            )
            response = requests.put(url, headers=publish_headers)

            if response.status_code == 200:
                published_entry = response.json()
                new_version = published_entry["sys"]["version"]
                self.logger.info(
                    f"✅ Top News container published successfully: v{current_version} → v{new_version}"
                )
                return True

            elif response.status_code == 422:
                self.logger.error(
                    f"❌ Top News container publishing failed - validation errors:"
                )
                error_data = response.json()
                if "details" in error_data and "errors" in error_data["details"]:
                    for error in error_data["details"]["errors"]:
                        error_name = error.get("name", "Unknown")
                        error_path = " → ".join(map(str, error.get("path", [])))
                        error_details = error.get("details", "No details")
                        self.logger.error(
                            f"      🚫 {error_name}: {error_path} - {error_details}"
                        )
                return False

            else:
                self.logger.error(
                    f"❌ Top News container publishing failed: {response.status_code}"
                )
                self.logger.error(f"   Response: {response.text}")
                return False

        except Exception as e:
            self.logger.error(f"❌ Error publishing Top News container: {e}")
            return False

    def publish_articles_batch(self, article_ids: List[str]) -> Dict[str, bool]:
        """Publish multiple articles with detailed error logging"""
        self.logger.info(
            f"📝 ENHANCED LOGGING: Publishing {len(article_ids)} articles..."
        )

        results = {}

        for i, article_id in enumerate(article_ids, 1):
            try:
                self.logger.info(
                    f"\n📝 PUBLISHING ARTICLE {i}/{len(article_ids)}: {article_id}"
                )

                # First, get the article to check its structure
                url = f"https://api.contentful.com/spaces/{self.space_id}/entries/{article_id}"
                response = requests.get(url, headers=self.headers)

                if response.status_code != 200:
                    self.logger.error(
                        f"❌ Failed to get article {article_id}: {response.status_code}"
                    )
                    self.logger.error(f"   Response: {response.text}")
                    results[article_id] = False
                    continue

                article = response.json()
                current_version = article["sys"]["version"]

                # Log article structure for debugging
                self.logger.info(f"   📋 Article structure check:")
                self.logger.info(f"      Version: {current_version}")
                self.logger.info(
                    f"      Available fields: {list(article['fields'].keys())}"
                )

                # Check for common required fields
                required_checks = {
                    "headline": article["fields"].get("headline"),
                    "publishedDate": article["fields"].get("publishedDate"),
                    "categories": article["fields"].get("categories"),
                }

                for field_name, field_value in required_checks.items():
                    if field_value:
                        self.logger.info(f"      ✅ {field_name}: Present")
                    else:
                        self.logger.warning(f"      ⚠️ {field_name}: Missing or empty")

                # Check if already published
                if "publishedVersion" in article["sys"]:
                    published_version = article["sys"]["publishedVersion"]
                    if published_version == current_version:
                        self.logger.info(
                            f"   ✅ Article already published (v{current_version})"
                        )
                        results[article_id] = True
                        continue
                    else:
                        self.logger.info(
                            f"   📝 Article needs republishing: published v{published_version} vs current v{current_version}"
                        )
                else:
                    self.logger.info(f"   📝 Article has never been published")

                # Attempt to publish
                publish_url = f"{url}/published"
                publish_headers = self.headers.copy()
                publish_headers["X-Contentful-Version"] = str(current_version)

                self.logger.info(f"   🚀 Attempting to publish v{current_version}...")
                response = requests.put(publish_url, headers=publish_headers)

                if response.status_code == 200:
                    published_article = response.json()
                    new_version = published_article["sys"]["version"]
                    self.logger.info(
                        f"   ✅ PUBLISH SUCCESS: v{current_version} → v{new_version}"
                    )
                    results[article_id] = True
                else:
                    self.logger.error(f"   ❌ PUBLISH FAILED: {response.status_code}")
                    results[article_id] = False

                    # Detailed error analysis
                    try:
                        error_data = response.json()
                        self.logger.error(f"   📋 Error details:")
                        if "message" in error_data:
                            self.logger.error(f"      Message: {error_data['message']}")
                        if (
                            "details" in error_data
                            and "errors" in error_data["details"]
                        ):
                            self.logger.error(f"      Validation errors:")
                            for error in error_data["details"]["errors"]:
                                error_name = error.get("name", "Unknown")
                                error_path = " → ".join(map(str, error.get("path", [])))
                                error_details = error.get("details", "No details")
                                self.logger.error(
                                    f"        🚫 {error_name}: {error_path}"
                                )
                                if error_details != "No details":
                                    self.logger.error(
                                        f"           Details: {error_details}"
                                    )
                    except:
                        self.logger.error(f"   📋 Raw error response: {response.text}")

            except Exception as e:
                self.logger.error(
                    f"❌ Exception while publishing article {article_id}: {e}"
                )
                results[article_id] = False

        successful = sum(1 for success in results.values() if success)
        total = len(results)
        self.logger.info(
            f"\n📝 PUBLISHING SUMMARY: {successful}/{total} articles published successfully"
        )

        if successful < total:
            failed_ids = [
                article_id for article_id, success in results.items() if not success
            ]
            self.logger.warning(f"   ❌ Failed articles: {failed_ids}")

        return results

    def _ensure_article_has_categories(self, article_id: str) -> bool:
        """Ensure article has categories field, assign General if missing"""
        self.logger.info(f"🔍 Checking categories for article: {article_id}")

        try:
            response = requests.get(
                f"https://api.contentful.com/spaces/{self.space_id}/entries/{article_id}",
                headers=self.headers,
            )

            if response.status_code != 200:
                self.logger.error(f"❌ Failed to get article: {response.status_code}")
                return False

            article = response.json()
            current_version = article["sys"]["version"]
            current_fields = article["fields"]

            categories_field = current_fields.get("categories")
            has_categories = False

            if categories_field:
                if isinstance(categories_field, dict) and "en-US" in categories_field:
                    categories_list = categories_field["en-US"]
                    if isinstance(categories_list, list) and len(categories_list) > 0:
                        has_categories = True
                        self.logger.info(
                            f"   ✅ Article already has categories: {categories_list}"
                        )

            if has_categories:
                return True

            self.logger.info(f"   ⚠️ Article missing categories, assigning General")

            updated_fields = current_fields.copy()
            updated_fields["categories"] = {"en-US": ["General"]}

            update_payload = {"fields": updated_fields}

            update_headers = self.headers.copy()
            update_headers["X-Contentful-Version"] = str(current_version)

            response = requests.put(
                f"https://api.contentful.com/spaces/{self.space_id}/entries/{article_id}",
                headers=update_headers,
                json=update_payload,
            )

            if response.status_code == 200:
                updated_article = response.json()
                new_version = updated_article["sys"]["version"]
                self.logger.info(f"   ✅ Categories assigned successfully!")
                self.logger.info(f"   📝 Version: {current_version} → {new_version}")
                return True
            else:
                self.logger.error(
                    f"   ❌ Failed to assign categories: {response.status_code}"
                )
                self.logger.error(f"   Response: {response.text}")
                return False

        except Exception as e:
            self.logger.error(f"   ❌ Error ensuring categories: {e}")
            return False


# --- 3. GAME LOGIC & AI GENERATION ---

# --- 3. GAME LOGIC & AI GENERATION ---

class QuizDirector:
    """Manages the AI generation of quiz content using Structured Outputs"""
    
    def __init__(self):
        self.model = genai.GenerativeModel("gemini-2.5-flash")
        
        # 1. Define the Strict Schema for Questions
        self.quiz_schema = {
            "type": "OBJECT",
            "properties": {
                "host_intro": {"type": "STRING", "description": "A short, 1-sentence intro in the host's specific persona voice."},
                "question": {"type": "STRING", "description": "The actual trivia question text."},
                "options": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"},
                    "description": "A list of exactly 4 options."
                },
                "correct_answer": {"type": "STRING", "description": "The correct option text, must match one of the options exactly."},
                "host_reaction_correct": {"type": "STRING", "description": "A short phrase the host says if the user gets it right."},
                "host_reaction_wrong": {"type": "STRING", "description": "A short phrase the host says if the user gets it wrong."}
            },
            "required": ["host_intro", "question", "options", "correct_answer", "host_reaction_correct", "host_reaction_wrong"]
        }

    def _get_persona_prompt(self, persona: str) -> str:
        if persona == "Professor":
            return "You are 'The Professor'. Tone: Dry, academic, precise. Focus on history and geography."
        elif persona == "Steve":
            return "You are a high-energy Game Show Host. Tone: Folksy, loud, uses slang like 'Survey says!'. Focus on people."
        elif persona == "Regis":
            return "You are a Dramatic Host. Tone: Suspenseful, intense short sentences. Focus on the high stakes."
        return "You are a neutral host."

    def generate_question(self, headline: str, persona: str, difficulty: str, topic: str = "General") -> dict:
        """Generates a question using Structured Output enforcement"""
        
        persona_instruction = self._get_persona_prompt(persona)
        
        prompt = f"""
        {persona_instruction}
        
        Generate a multiple-choice trivia question based on this news headline: 
        "{headline}"
        
        DIFFICULTY: {difficulty} (Easy = basic fact check, Medium = context, Hard = related history/trivia)
        TOPIC: {topic}
        
        Ensure 'options' contains exactly 4 strings.
        Ensure 'correct_answer' is exactly equal to one of the strings in 'options'.
        """
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=self.quiz_schema,  # <--- THIS ENFORCES STRUCTURE
                    temperature=0.7 
                )
            )
            return json.loads(response.text)

        except Exception as e:
            print(f"\n❌ AI GENERATION FAILED: {e}")
            # Fallback is only needed if the API is down/blocked
            return {
                "host_intro": "We're having some trouble with the satellite feed, but let's press on!",
                "question": f"This story is about: {headline}. Which category does it fit best?",
                "options": ["Sports", "Politics", "Entertainment", "Technology"],
                "correct_answer": "Politics",
                "host_reaction_correct": "You got it!",
                "host_reaction_wrong": "Not quite."
            }

    def rewrite_question(self, current_q_data: dict, new_persona: str) -> dict:
        """Rewrites existing question in new persona using Structured Output"""
        persona_instruction = self._get_persona_prompt(new_persona)
        
        prompt = f"""
        {persona_instruction}
        Rewrite this question data to match your specific voice/tone. 
        Keep the core facts, the question logic, and the correct answer EXACTLY the same.
        Only change the phrasing of the intro, question text, and reactions.
        
        INPUT DATA: {json.dumps(current_q_data)}
        """
        try:
            response = self.model.generate_content(
                prompt, 
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=self.quiz_schema # <--- ENFORCED HERE TOO
                )
            )
            return json.loads(response.text)
        except:
            return current_q_data

    def get_vinny_response(self, question: str, correct_answer: str, options: List[str]) -> str:
        """Generates Cousin Vinny's text message"""
        # Logic: 70% chance to pick the right answer, 30% random wrong answer
        is_correct = random.random() < 0.70
        
        if is_correct:
            target_answer = correct_answer
        else:
            # Filter out the correct answer to pick a wrong one
            wrong_options = [opt for opt in options if opt != correct_answer]
            if wrong_options:
                target_answer = random.choice(wrong_options)
            else:
                target_answer = "I honestly dunno"

        # Vinny doesn't need complex schema, just a string
        prompt = f"""
        You are 'Cousin Vinny' from New York. You are texting your cousin who is on a game show.
        Write a VERY short text message (under 15 words) telling them the answer is: "{target_answer}".
        Use slang, maybe a typo. Be confident.
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except:
            return f"Yo! It's definitely {target_answer}. Trust me!"

    def generate_topic_choices(self, remaining_headlines: List[str]) -> Dict[str, str]:
        return {
            "Politics & Power": "politics",
            "World & Money": "international",
            "Wildcard / Weird": "general"
        }
    
# --- 4. STATE MANAGEMENT ---


def init_session():
    if "game_state" not in st.session_state:
        st.session_state.game_state = "START"  # START, PLAYING, END
    if "deck" not in st.session_state:
        st.session_state.deck = []  # List of story objects
    if "current_q_index" not in st.session_state:
        st.session_state.current_q_index = 0
    if "current_q_data" not in st.session_state:
        st.session_state.current_q_data = None
    if "score" not in st.session_state:
        st.session_state.score = 0
    if "lifelines" not in st.session_state:
        st.session_state.lifelines = {
            "50_50": True,
            "phone_vinny": True,
            "switch_host": True,
            "choose_topic": True,
        }
    if "disabled_options" not in st.session_state:
        st.session_state.disabled_options = []  # For 50/50
    if "vinny_msg" not in st.session_state:
        st.session_state.vinny_msg = None
    if "assigned_personas" not in st.session_state:
        # Pre-assign random personas to ensure 1/3rd split
        personas = ["Professor"] * 4 + ["Steve"] * 3 + ["Regis"] * 3
        random.shuffle(personas)
        st.session_state.assigned_personas = personas


def reset_game():
    st.session_state.game_state = "START"
    st.session_state.score = 0
    st.session_state.current_q_index = 0
    st.session_state.deck = []
    st.session_state.lifelines = {k: True for k in st.session_state.lifelines}
    st.session_state.vinny_msg = None
    st.session_state.disabled_options = []


# --- 5. MAIN APPLICATION LOGIC ---


def main():
    init_session()

    # Header
    c1, c2 = st.columns([3, 1])
    with c1:
        st.title("📺 PRIME TIME QUIZ NIGHT")
    with c2:
        st.markdown("<div class='live-indicator'>● LIVE</div>", unsafe_allow_html=True)

    # --- SCREEN: START ---
    if st.session_state.game_state == "START":
        st.markdown("### 🌍 Aggregating News from All Major Networks...")
        st.info(
            "We are getting the latest trending stories to generate the quiz questions. Click 'GO LIVE' when ready!"
        )

        if st.button("🎥 GO LIVE (Start Show)", type="primary"):
            with st.spinner(
                "Connecting to Satellites... Extracting Headlines... Clustering Topics..."
            ):

                # --- USE YOUR REAL CLASS HERE ---
                # If you pasted the real code, verify this class name matches
                try:
                    # Use the Unified System from Code A
                    news_system = UnifiedNewsClusteringSystem()
                    clusters = news_system.get_top_trending_stories(count=10)

                    if not clusters:
                        st.error("No news found. Are scrapers blocked?")
                    else:
                        st.session_state.deck = clusters
                        st.session_state.game_state = "PLAYING"
                        st.rerun()
                except NameError:
                    st.error(
                        "⚠️ Critical Error: Please paste the Code A classes into the `app.py` file where indicated! The scraper classes are missing."
                    )
                except Exception as e:
                    st.error(f"System Failure: {e}")

    # --- SCREEN: PLAYING ---
    elif st.session_state.game_state == "PLAYING":

        director = QuizDirector()

        # Check Game Over
        if st.session_state.current_q_index >= 10:
            st.session_state.game_state = "END"
            st.rerun()

        # Get Current Story & Metadata
        current_story = st.session_state.deck[st.session_state.current_q_index]
        current_persona = st.session_state.assigned_personas[
            st.session_state.current_q_index
        ]

        # Determine Difficulty (Fixed progression)
        idx = st.session_state.current_q_index
        difficulty = "Easy" if idx < 4 else "Medium" if idx < 8 else "Hard"

        # Generate Question (Lazy Load - generate only if not already in state for this index)
        # Note: We store q_data in session_state so it persists during button clicks (lifelines)
        if st.session_state.current_q_data is None:
            with st.spinner(f"{current_persona} is reading the cue cards..."):
                q_data = director.generate_question(
                    current_story.representative_headline,
                    current_persona,
                    difficulty,
                    topic=current_story.category,
                )
                st.session_state.current_q_data = q_data
                # Reset per-question states
                st.session_state.disabled_options = []
                st.session_state.vinny_msg = None

        q_data = st.session_state.current_q_data

        # --- SIDEBAR: LIFELINES ---
        with st.sidebar:
            st.subheader("🆘 Lifelines")

            # 1. 50/50
            if st.session_state.lifelines["50_50"]:
                if st.button("✂️ 50/50", key="ll_5050"):
                    st.session_state.lifelines["50_50"] = False
                    # Identify wrong answers
                    wrong_opts = [
                        o for o in q_data["options"] if o != q_data["correct_answer"]
                    ]
                    # Disable 2 random wrong ones
                    st.session_state.disabled_options = random.sample(
                        wrong_opts, min(2, len(wrong_opts))
                    )
                    st.rerun()
            else:
                st.button("✂️ 50/50 (Used)", disabled=True)

            # 2. Phone Vinny
            if st.session_state.lifelines["phone_vinny"]:
                if st.button("📞 Text Vinny", key="ll_vinny"):
                    st.session_state.lifelines["phone_vinny"] = False
                    msg = director.get_vinny_response(
                        q_data["question"], q_data["correct_answer"], q_data["options"]
                    )
                    st.session_state.vinny_msg = msg
                    st.rerun()
            else:
                st.button("📞 Vinny (Used)", disabled=True)

            # 3. Switch Host
            if st.session_state.lifelines["switch_host"]:
                if st.button("🔄 Switch Host", key="ll_switch"):
                    st.session_state.lifelines["switch_host"] = False
                    # Pick new persona
                    avail_personas = [
                        p
                        for p in ["Professor", "Steve", "Regis"]
                        if p != current_persona
                    ]
                    new_p = random.choice(avail_personas)
                    # Rewrite
                    with st.spinner("Swapping microphones..."):
                        new_q_data = director.rewrite_question(q_data, new_p)
                        st.session_state.current_q_data = new_q_data
                        # Update assigned persona list for this index
                        st.session_state.assigned_personas[
                            st.session_state.current_q_index
                        ] = new_p
                    st.rerun()
            else:
                st.button("🔄 Switch (Used)", disabled=True)

            # 4. Choose Topic (Commercial Break)
            if st.session_state.lifelines["choose_topic"]:
                if st.button("🎯 Choose Topic", key="ll_topic"):
                    # This logic needs to happen AFTER this question is answered, theoretically.
                    # OR, we skip this question entirely?
                    # Let's implement: "Skip this, pick next topic".
                    st.session_state.lifelines["choose_topic"] = False
                    st.session_state.show_topic_picker = True
                    st.rerun()
            else:
                st.button("🎯 Topic (Used)", disabled=True)

        # --- MAIN STAGE ---

        # Progress Bar
        st.progress((idx + 1) / 10)
        st.caption(f"Question {idx + 1}/10 • Difficulty: {difficulty}")

        # Host Badge
        badge_class = f"badge-{current_persona.lower()}"
        st.markdown(
            f"<span class='{badge_class}'>{current_persona}</span>",
            unsafe_allow_html=True,
        )

        # Intro Dialogue
        st.markdown(f"_{q_data['host_intro']}_")

        # Question Card
        st.markdown(
            f"""
        <div class='tv-card'>
            <h3>{q_data['question']}</h3>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Vinny Overlay
        if st.session_state.vinny_msg:
            st.info(f"📱 **Cousin Vinny says:** {st.session_state.vinny_msg}")

        # Answer Options
        options = q_data["options"]

        c1, c2 = st.columns(2)
        for i, opt in enumerate(options):
            col = c1 if i % 2 == 0 else c2

            # 50/50 Logic: Hide button if disabled
            if opt in st.session_state.disabled_options:
                col.button(f"❌ {opt}", disabled=True, key=f"opt_{i}")
            else:
                if col.button(opt, key=f"opt_{i}"):
                    # CHECK ANSWER
                    if opt == q_data["correct_answer"]:
                        st.toast(f"✅ {q_data['host_reaction_correct']}")
                        st.balloons()
                        st.session_state.score += 1
                        time.sleep(2)
                    else:
                        st.toast(f"❌ {q_data['host_reaction_wrong']}")
                        st.error(f"Wrong! Correct was: {q_data['correct_answer']}")
                        time.sleep(2)

                    # Next Question Setup
                    st.session_state.current_q_index += 1
                    st.session_state.current_q_data = (
                        None  # Clear so next one generates
                    )
                    st.rerun()

        # Topic Picker Overlay (If activated)
        if st.session_state.get("show_topic_picker", False):
            st.markdown("---")
            st.warning("⏸️ COMMERCIAL BREAK! Pick the next topic:")
            cols = st.columns(3)
            if cols[0].button("Politics & Power"):
                # In V2: Reorder deck to put politics next. V1: Just proceed
                st.session_state.show_topic_picker = False
                st.rerun()
            if cols[1].button("World & Money"):
                st.session_state.show_topic_picker = False
                st.rerun()
            if cols[2].button("Weird / Wildcard"):
                st.session_state.show_topic_picker = False
                st.rerun()

    # --- SCREEN: END ---
    elif st.session_state.game_state == "END":
        st.markdown(
            "<div style='text-align: center; padding-top: 50px;'>",
            unsafe_allow_html=True,
        )
        st.title("🎬 THAT'S A WRAP!")
        st.markdown(
            f"<h1>Final Score: {st.session_state.score} / 10</h1>",
            unsafe_allow_html=True,
        )

        if st.session_state.score > 7:
            st.balloons()
            st.markdown("### 🏆 LEGENDARY STATUS!")
        elif st.session_state.score > 4:
            st.markdown("### 👍 NOT BAD, KID.")
        else:
            st.markdown("### 📉 BACK TO SCHOOL FOR YOU.")

        if st.button("🔄 Play Again (New News)"):
            reset_game()
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
