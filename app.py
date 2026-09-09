import streamlit as st
import io
import re
import base64
import requests
from gtts import gTTS
from huggingface_hub import InferenceClient

# Config Halaman
st.set_page_config(
    page_title="日本語会話 - Speech-to-Speech Kaiwa AI",
    page_icon="🎙️",
    layout="centered"
)

st.title("🎙️ Speech-to-Speech Japanese Kaiwa")
st.caption("Ngobrol bahasa Jepang langsung pakai suara dengan penjelasan Bahasa Indonesia!")

# --- SIDEBAR CONFIG ---
st.sidebar.header("⚙️ Konfigurasi & Status")

SECRET_TOKEN = st.secrets.get("HF_TOKEN", "")

if SECRET_TOKEN:
    HF_TOKEN = SECRET_TOKEN
    st.sidebar.success("✅ Token dari Secrets Terpasang")
else:
    HF_TOKEN = st.sidebar.text_input(
        "Hugging Face API Token",
        type="password",
        help="Masukkan Access Token Hugging Face kamu (berawalan hf_). "
             "Pastikan token punya izin 'Make calls to Inference Providers'."
    )
    if HF_TOKEN:
        st.sidebar.success("✅ Token Manual Terpasang")
    else:
        st.sidebar.warning("⚠️ Masukkan Token HF untuk melanjutkan")

st.sidebar.markdown("---")
st.sidebar.subheader("💡 Cara Pakai:")
st.sidebar.write("1. Rekam suara Anda menggunakan perekam di bawah.")
st.sidebar.write("2. Dengarkan hasil rekaman, lalu klik tombol Kirim.")
st.sidebar.write("3. AI akan menjawab dengan format Respond + Question, plus suara balasan.")

# --- MODEL / PROVIDER CONFIG ---
STT_MODEL = "openai/whisper-large-v3-turbo"
LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
LLM_PROVIDER = "featherless-ai"


# Function: Speech-to-Text via Router Endpoint
# Dikirim sebagai JSON (base64) -> supaya bisa memaksa language="japanese"
# lewat generate_kwargs. Kirim raw bytes saja TIDAK bisa menyisipkan parameter
# ini, sehingga Whisper sering auto-detect bahasa dengan salah (mis. terbaca Korea).
def transcribe_audio(audio_bytes):
    if not HF_TOKEN:
        return {"error": "Token Hugging Face belum terpasang."}

    api_url = f"https://router.huggingface.co/hf-inference/models/{STT_MODEL}"

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "inputs": base64.b64encode(audio_bytes).decode("utf-8"),
        "parameters": {
            "generate_kwargs": {
                "language": "japanese",
                "task": "transcribe"
            }
        }
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)

        if response.status_code == 200:
            result = response.json()
            if isinstance(result, dict) and "text" in result:
                return {"text": result["text"]}
            elif isinstance(result, list) and len(result) > 0 and "text" in result[0]:
                return {"text": result[0]["text"]}
            return {"text": str(result)}
        else:
            try:
                err_data = response.json()
                error_msg = err_data.get("error", f"HTTP {response.status_code}")
            except Exception:
                error_msg = f"HTTP Error {response.status_code}: {response.text}"
            return {"error": str(error_msg)}
    except Exception as e:
        return {"error": str(e)}


# Function: LLM Response
# System prompt dipaksa strict 2 baris: "Respond :" dan "Question :"
def generate_response(messages):
    if not HF_TOKEN:
        return "Error: Token Hugging Face belum terpasang."

    try:
        client = InferenceClient(provider=LLM_PROVIDER, api_key=HF_TOKEN)

        system_prompt = (
            "You are a friendly Japanese conversation partner (Kaiwa AI) for language learners. "
            "You MUST reply in EXACTLY this two-line format, nothing else, no extra explanation:\n"
            "Respond : <a short natural reaction in Japanese, written in hiragana/katakana/kanji>\n"
            "Question : <one short natural follow-up question in Japanese, written in hiragana/katakana/kanji>\n"
            "Never add romaji, translation, or any other text outside these two lines."
        )

        formatted_messages = [{"role": "system", "content": system_prompt}]
        for msg in messages:
            if msg["role"] != "system":
                formatted_messages.append({"role": msg["role"], "content": msg["content"]})

        response = client.chat_completion(
            messages=formatted_messages,
            model=LLM_MODEL,
            max_tokens=150,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error LLM: {str(e)}"


# Function: Generate Audio Autoplay HTML (gTTS)
# Ambil isi setelah label "Respond :" dan "Question :" (Jepang asli, bukan label-nya)
# supaya gTTS mengucapkan bahasa Jepang yang benar, lalu gabung jadi satu audio.
def play_audio_autoplay(text_ja):
    try:
        spoken_parts = []
        for line in text_ja.split("\n"):
            line = line.strip()
            if ":" in line:
                _, content = line.split(":", 1)
                content = content.strip()
                if content:
                    spoken_parts.append(content)

        spoken_text = "。".join(spoken_parts) if spoken_parts else text_ja

        tts = gTTS(text=spoken_text, lang="ja")
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
        {"role": "assistant", "content": "Respond : こんにちは！\nQuestion : 今日は元気ですか？"}
    ]

# Tampilkan Chat History
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

st.divider()

# Kontrol Perekaman Suara (manual, sebelum dikirim)
st.markdown("### 🗣️ Rekam Suara Anda:")
audio_file = st.audio_input("Gunakan mikrofon Anda untuk merekam percakapan")

if audio_file is not None:
    audio_bytes = audio_file.getvalue()

    if st.button("🚀 Kirim Suara ke AI", type="primary"):
        if not HF_TOKEN:
            st.error("⚠️ Masukkan Hugging Face Token di sidebar terlebih dahulu!")
        else:
            with st.spinner("🎙️ Menerjemahkan suara Anda..."):
                stt_result = transcribe_audio(audio_bytes)

                if isinstance(stt_result, dict) and "error" in stt_result:
                    st.error(f"⚠️ {stt_result['error']}")
                elif isinstance(stt_result, dict) and stt_result.get("text"):
                    user_speech = stt_result["text"].strip()

                    if user_speech:
                        st.session_state.messages.append({"role": "user", "content": user_speech})
                        with st.chat_message("user"):
                            st.write(f"🗣️ *\"{user_speech}\"*")

                        with st.spinner("🤖 AI sedang menyusun jawaban..."):
                            bot_reply = generate_response(st.session_state.messages)

                        st.session_state.messages.append({"role": "assistant", "content": bot_reply})

                        with st.chat_message("assistant"):
                            st.write(bot_reply)
                            play_audio_autoplay(bot_reply)
                        # Catatan: TIDAK memanggil st.rerun() di sini -
                        # itu penyebab audio langsung mati sebelum sempat diputar.
                    else:
                        st.warning("Suara tidak terdengar jelas. Coba rekam ulang.")
                else:
                    st.warning("Gagal memproses suara. Coba rekam ulang.")
