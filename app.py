import streamlit as st
import io
import base64
from gTTS import gTTS
from audio_recorder_streamlit import audio_recorder
from huggingface_hub import InferenceClient

# Config Halaman
st.set_page_config(
    page_title="日本語会話 - Speech-to-Speech Kaiwa AI",
    page_icon="🎙️",
    layout="centered"
)

st.title("🎙️ Speech-to-Speech Japanese Kaiwa")
st.caption("Ngobrol bahasa Jepang langsung pakai suara secara real-time!")

# Mengambil HF Token dari Secrets / Hardcode Fallback
HF_TOKEN = st.secrets.get("HF_TOKEN", "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")

st.sidebar.header("⚙️ Status App")
st.sidebar.success("✅ Hugging Face Token Terpasang")

# Inisialisasi Client SDK Resmi Hugging Face
client = InferenceClient(provider="hf-inference", api_key=HF_TOKEN)

# Model
STT_MODEL = "openai/whisper-large-v3-turbo"
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Function: Speech-to-Text (Whisper via SDK)
def transcribe_audio(audio_bytes):
    try:
        res = client.automatic_speech_recognition(
            audio=audio_bytes,
            model=STT_MODEL
        )
        return {"text": res}
    except Exception as e:
        return {"error": str(e)}

# Function: LLM Response (Qwen via SDK Chat Completion)
def generate_response(messages):
    system_prompt = (
        "You are a friendly, encouraging Japanese conversation partner (Kaiwa AI). "
        "Always respond naturally in Japanese suitable for language learners. "
        "On a new line below the Japanese text, provide Romaji and English translation for learning."
    )
    
    formatted_messages = [{"role": "system", "content": system_prompt}]
    for msg in messages:
        if msg["role"] != "system":
            formatted_messages.append({"role": msg["role"], "content": msg["content"]})

    try:
        response = client.chat_completion(
            messages=formatted_messages,
            model=LLM_MODEL,
            max_tokens=256,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error LLM: {str(e)}"

# Function: Generate Audio Autoplay HTML
def play_audio_autoplay(text_ja):
    try:
        ja_sentence = text_ja.split("\n")[0]
        tts = gTTS(text=ja_sentence, lang="ja")
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        b64_audio = base64.b64encode(fp.read()).decode("utf-8")
        audio_html = f"""
            <audio autoplay style="display:none;">
                <source src="data:audio/mp3;base64,{b64_audio}" type="audio/mp3">
            </audio>
        """
        st.components.v1.html(audio_html, height=0)
    except Exception as e:
        st.warning(f"Gagal memutar audio: {e}")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "System Initialized"},
        {"role": "assistant", "content": "こんにちは！一緒に日本語を練習しましょう！\n(Konnichiwa! Issho ni Nihongo wo renshuu shimashou! / Hello! Let's practice Japanese together!)"}
    ]

# Tampilkan Chat History
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

st.divider()

# Audio Recorder Container
st.markdown("### 🗣️ Bicara Sekarang:")
audio_bytes = audio_recorder(
    text="Klik untuk Merekam",
    recording_color="#e74c3c",
    neutral_color="#2ecc71",
    icon_size="2x"
)

# Process Audio
if audio_bytes:
    with st.spinner("🎙️ Mengubah suara ke teks..."):
        stt_result = transcribe_audio(audio_bytes)
        
        if isinstance(stt_result, dict) and "error" in stt_result:
            st.error(f"⚠️ {stt_result['error']}")
        elif isinstance(stt_result, dict) and "text" in stt_result and stt_result["text"]:
            user_speech = stt_result["text"]
            if hasattr(user_speech, "text"):
                user_speech = user_speech.text

            user_speech = str(user_speech).strip()
            
            if user_speech:
                st.session_state.messages.append({"role": "user", "content": user_speech})
                with st.chat_message("user"):
                    st.write(f"🗣️ *\"{user_speech}\"*")

                with st.spinner("🤖 AI sedang memikirkan balasan..."):
                    bot_reply = generate_response(st.session_state.messages)

                st.session_state.messages.append({"role": "assistant", "content": bot_reply})
                with st.chat_message("assistant"):
                    st.write(bot_reply)
                    play_audio_autoplay(bot_reply)
            else:
                st.warning("Suara tidak terdeteksi. Coba rekam ulang.")
