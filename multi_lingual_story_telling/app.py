"""
Multi-Lingual Story Telling
============================
A Streamlit web application that:
  1. Accepts text input directly or via file upload (TXT, PDF, CSV, Excel)
  2. Translates the text into a user-selected language using OpenAI GPT-3.5-turbo
  3. Converts the translated text to speech using Google Text-to-Speech (gTTS)
  4. Lets the user play and download the audio file
"""

import os
import io
import tempfile

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from gtts import gTTS
import PyPDF2
import pandas as pd

# ── Load environment variables ────────────────────────────────────────────────
load_dotenv()

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Lingual Story Telling",
    page_icon="🌐",
    layout="centered",
)

# ── Supported languages (language name → gTTS language code) ─────────────────
LANGUAGES = {
    "Afrikaans"          : "af",
    "Arabic"             : "ar",
    "Bengali"            : "bn",
    "Chinese (Simplified)": "zh-CN",
    "Chinese (Traditional)": "zh-TW",
    "Czech"              : "cs",
    "Danish"             : "da",
    "Dutch"              : "nl",
    "English"            : "en",
    "Finnish"            : "fi",
    "French"             : "fr",
    "German"             : "de",
    "Greek"              : "el",
    "Gujarati"           : "gu",
    "Hebrew"             : "iw",
    "Hindi"              : "hi",
    "Hungarian"          : "hu",
    "Indonesian"         : "id",
    "Italian"            : "it",
    "Japanese"           : "ja",
    "Kannada"            : "kn",
    "Korean"             : "ko",
    "Latvian"            : "lv",
    "Lithuanian"         : "lt",
    "Malay"              : "ms",
    "Malayalam"          : "ml",
    "Marathi"            : "mr",
    "Nepali"             : "ne",
    "Norwegian"          : "no",
    "Polish"             : "pl",
    "Portuguese"         : "pt",
    "Punjabi"            : "pa",
    "Romanian"           : "ro",
    "Russian"            : "ru",
    "Serbian"            : "sr",
    "Sinhala"            : "si",
    "Slovak"             : "sk",
    "Spanish"            : "es",
    "Swahili"            : "sw",
    "Swedish"            : "sv",
    "Tamil"              : "ta",
    "Telugu"             : "te",
    "Thai"               : "th",
    "Turkish"            : "tr",
    "Ukrainian"          : "uk",
    "Urdu"               : "ur",
    "Vietnamese"         : "vi",
}

# ── Helper functions ──────────────────────────────────────────────────────────

def get_openai_client() -> OpenAI:
    """Return an OpenAI client using key from env or session state."""
    api_key = st.session_state.get("openai_api_key") or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        st.error("OpenAI API key is missing. Enter it in the sidebar.")
        st.stop()
    return OpenAI(api_key=api_key)


def translate_text(text: str, target_language: str) -> str:
    """Translate *text* into *target_language* using GPT-3.5-turbo."""
    client = get_openai_client()
    prompt = (
        f"Translate the following text into {target_language}. "
        f"Return only the translated text, without any explanation or extra commentary.\n\n"
        f"{text}"
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a professional multilingual translator. "
                    "Translate text accurately while preserving tone, style, and meaning."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=2000,
    )
    return response.choices[0].message.content.strip()


def text_to_speech(text: str, lang_code: str) -> bytes:
    """Convert *text* to MP3 bytes using gTTS."""
    tts = gTTS(text=text, lang=lang_code, slow=False)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()


def extract_text_from_file(uploaded_file) -> str:
    """Extract plain text from TXT, PDF, CSV, or Excel uploads."""
    name = uploaded_file.name.lower()

    if name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")

    if name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(uploaded_file)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).strip()

    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        return df.to_string(index=False)

    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
        return df.to_string(index=False)

    return ""


# ── UI ────────────────────────────────────────────────────────────────────────

# Sidebar – API key
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Your key is never stored permanently.",
        value=st.session_state.get("openai_api_key", ""),
    )
    if api_key_input:
        st.session_state["openai_api_key"] = api_key_input

    st.markdown("---")
    st.markdown(
        "**Supported file types for upload:**\n"
        "- 📄 `.txt` – Plain text\n"
        "- 📕 `.pdf` – PDF document\n"
        "- 📊 `.csv` – CSV file\n"
        "- 📗 `.xlsx` / `.xls` – Excel\n"
    )
    st.markdown("---")
    st.caption("Powered by OpenAI GPT-3.5-turbo & gTTS")

# Main title
st.title("🌐 Multi-Lingual Story Telling")
st.markdown(
    "Translate text into any language and listen to it as speech. "
    "Type your text below or upload a file."
)
st.markdown("---")

# ── Step 1: Text Input ────────────────────────────────────────────────────────
st.subheader("Step 1 – Enter or Upload Text")

input_tab, upload_tab = st.tabs(["✏️ Type Text", "📂 Upload File"])

input_text = ""

with input_tab:
    input_text_area = st.text_area(
        "Enter your text here:",
        height=200,
        placeholder="Once upon a time, in a land far away...",
    )
    if input_text_area.strip():
        input_text = input_text_area.strip()

with upload_tab:
    uploaded_file = st.file_uploader(
        "Upload a TXT, PDF, CSV, or Excel file",
        type=["txt", "pdf", "csv", "xlsx", "xls"],
    )
    if uploaded_file:
        with st.spinner("Extracting text from file..."):
            extracted = extract_text_from_file(uploaded_file)
        if extracted.strip():
            input_text = extracted.strip()
            st.success(f"Text extracted successfully ({len(input_text)} characters).")
            with st.expander("Preview extracted text"):
                st.text(input_text[:2000] + ("..." if len(input_text) > 2000 else ""))
        else:
            st.error("Could not extract text from the uploaded file.")

# Show character count
if input_text:
    st.caption(f"📝 {len(input_text)} characters ready for translation.")

st.markdown("---")

# ── Step 2: Language Selection ────────────────────────────────────────────────
st.subheader("Step 2 – Select Target Language")

col1, col2 = st.columns([2, 1])
with col1:
    selected_language = st.selectbox(
        "Translate into:",
        options=list(LANGUAGES.keys()),
        index=list(LANGUAGES.keys()).index("French"),
    )
with col2:
    lang_code = LANGUAGES[selected_language]
    st.metric("Language Code", lang_code)

st.markdown("---")

# ── Step 3: Translate & Convert ───────────────────────────────────────────────
st.subheader("Step 3 – Translate & Generate Audio")

translate_btn = st.button(
    "🚀 Translate & Convert to Speech",
    type="primary",
    disabled=not bool(input_text),
    use_container_width=True,
)

if not input_text:
    st.info("Enter or upload text in Step 1 to enable translation.")

if translate_btn and input_text:
    # Translation
    with st.spinner(f"Translating to {selected_language} using GPT-3.5-turbo..."):
        try:
            translated_text = translate_text(input_text, selected_language)
        except Exception as e:
            st.error(f"Translation failed: {e}")
            st.stop()

    st.success("Translation complete!")

    # Display original and translated side by side
    col_orig, col_trans = st.columns(2)
    with col_orig:
        st.markdown("**Original Text**")
        st.text_area("", input_text, height=200, key="orig_display", disabled=True)
    with col_trans:
        st.markdown(f"**Translated Text ({selected_language})**")
        st.text_area("", translated_text, height=200, key="trans_display", disabled=True)

    # Download translated text
    st.download_button(
        label="📥 Download Translated Text (.txt)",
        data=translated_text.encode("utf-8"),
        file_name=f"translation_{selected_language.lower().replace(' ', '_')}.txt",
        mime="text/plain",
    )

    st.markdown("---")

    # Text-to-Speech
    with st.spinner(f"Converting to speech ({selected_language})..."):
        try:
            audio_bytes = text_to_speech(translated_text, lang_code)
        except Exception as e:
            st.error(f"Text-to-speech conversion failed: {e}")
            st.stop()

    st.success("Audio generated!")

    # Audio player
    st.audio(audio_bytes, format="audio/mp3")

    # Download audio
    st.download_button(
        label="🔊 Download Audio (.mp3)",
        data=audio_bytes,
        file_name=f"audio_{selected_language.lower().replace(' ', '_')}.mp3",
        mime="audio/mpeg",
    )

    st.markdown("---")
    st.balloons()
    st.markdown(
        f"✅ **Done!** Your text has been translated into **{selected_language}** "
        f"and converted to speech. Use the buttons above to download."
    )
