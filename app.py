import streamlit as st
import requests
import io
import base64
from gtts import gTTS
from audio_recorder_streamlit import audio_recorder

# Config Halaman
st.set_page_config(
    page_title="日本語会話 - Speech-to-Speech Kaiwa AI",
    page_icon="🎙️",
    layout="centered"
)

st.title("🎙️ Speech-to-Speech Japanese Kaiwa")
st.caption("Ngobrol bahasa Jepang langsung pakai suara secara real-time!")

# Sidebar Config
st.sidebar.header("⚙️ Pengaturan")
HF_TOKEN = st.sidebar.text_input("Hugging Face API Token", type="password")

st.sidebar.markdown("""
---
### 💡 Cara Pakai:
1. Masukkan **Hugging Face Token** di sidebar.
2. Klik ikon **Mikrofon** untuk mulai merekam suara.
3. Bicara dalam Bahasa Jepang (misal: *Konnichiwa, o-genki desu ka?*).
4. Klik ikon **Stop** untuk mengirim.
5. AI akan menjawab dalam teks & memutar suara balasan **otomatis**!
""")

# Endpoints Model Hugging Face
STT_MODEL = "openai/whisper-large-v3-turbo"
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Function: Speech-to-Text (Whisper API)
def transcribe_audio(audio_bytes):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    api_url = f"https://api-inference.huggingface.co/models/{STT_MODEL}"
    response = requests.post(api_url, headers=headers, data=audio_bytes)
    return response.json()

# Function: LLM Response (Qwen 2.5 API)
def generate_response(messages):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    api_url = f"https://api-inference.huggingface.co/models/{LLM_MODEL}"
    
    # Prompt System
    prompt = (
        "<|im_start|>system\n"
        "You are a friendly, encouraging Japanese conversation partner (Kaiwa AI). "
        "Always respond naturally in Japanese suitable for language learners. "
        "On a new line below the Japanese text, provide Romaji and English translation for learning.<|im_end|>\n"
    )
    
    for msg in messages:
        if msg["role"] != "system":
            prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 256,
            "temperature": 0.7,
            "return_full_text": False
        }
    }
    response = requests.post(api_url, headers=headers, json=payload)
    res = response.json()
    
    if isinstance(res, list) and len(res) > 0:
        raw_text = res[0].get("generated_text", "")
        clean_text = raw_text.split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "").strip()
        return clean_text
    return "すみません、もう一度言っていただけますか？\n(Sumimasen, mou ichido itte itadakemasu ka? / Sorry, could you say that again?)"

# Function: Generate Audio Autoplay HTML
def play_audio_autoplay(text_ja):
    # Ambil baris pertama (teks Jepang murni) untuk TTS
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

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "System Initialized"},
        {"role": "assistant", "content": "こんにちは！一緒に日本語を練習しましょう！\n(Konnichiwa! Issho ni Nihongo wo renshuu shimashou! / Hello! Let's practice Japanese together!)"}
    ]

# Display Chat History
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

# Process Audio Input
if audio_bytes:
    if not HF_TOKEN:
        st.error("⚠️ Masukkan Hugging Face Token di sidebar terlebih dahulu!")
    else:
        with st.spinner("🎙️ Mengubah suara ke teks (Whisper API)..."):
            stt_result = transcribe_audio(audio_bytes)
            
            if "text" in stt_result and stt_result["text"].strip():
                user_speech = stt_result["text"].strip()
                
                # Tambahkan ke History & Tampilkan
                st.session_state.messages.append({"role": "user", "content": user_speech})
                with st.chat_message("user"):
                    st.write(f"🗣️ *\"{user_speech}\"*")

                # Generate Balasan AI
                with st.spinner("🤖 AI sedang memikirkan balasan..."):
                    bot_reply = generate_response(st.session_state.messages)

                # Simpan Balasan AI & Tampilkan
                st.session_state.messages.append({"role": "assistant", "content": bot_reply})
                with st.chat_message("assistant"):
                    st.write(bot_reply)
                    
                    # Putar Audio Secara Otomatis (S2S Experience)
                    play_audio_autoplay(bot_reply)

            else:
                st.warning("Suara tidak terdengar jelas atau gagal dikonversi. Coba rekam ulang.")
