import streamlit as st
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
import tempfile
import pydub

# Initialize the recognizer
recognizer = sr.Recognizer()

def transcribe_audio(audio_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as temp_audio_file:
        temp_audio_file.write(audio_bytes)
        temp_audio_file.close()

        # Convert .ogg to .wav using pydub
        audio = pydub.AudioSegment.from_ogg(temp_audio_file.name)
        wav_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        audio.export(wav_temp_file.name, format="wav")
        wav_temp_file.close()

        # Recognize the speech in the wav file
        with sr.AudioFile(wav_temp_file.name) as source:
            audio = recognizer.record(source)
            try:
                # Recognize speech using Google Web Speech API
                text = recognizer.recognize_google(audio)
                return text
            except sr.UnknownValueError:
                return "Could not understand the audio."
            except sr.RequestError:
                return "Could not request results from Google Speech Recognition service."

# Streamlit UI
st.set_page_config(page_title="Voice Input Test", page_icon="🎙️")
st.title("🎙️ Voice-Enabled Chatbot")

# Record voice

voice_text = mic_recorder(start_prompt="🎤 Speak", stop_prompt="🛑 Stop", just_once=True)

if voice_text and 'bytes' in voice_text:
    # Convert the raw byte data to text
    user_input = transcribe_audio(voice_text['bytes'])
    st.write(f"Voice input: {user_input}")
else:
    st.write("No voice input detected.")
