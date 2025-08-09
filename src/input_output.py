import asyncio
from ai_handler import AiHandler
from media_pipe_handler import MediaPipeHandler
import speech_recognition as sr
import edge_tts as tts
import pyaudio
import io
from pydub import AudioSegment

AudioSegment.converter = "ffmpeg"    # or full path if you want
AudioSegment.ffprobe = "ffprobe"

recognizer = sr.Recognizer()
listen_and_speak = True
VOICE = "en-US-AndrewNeural"
api = AiHandler()
mp_handler = MediaPipeHandler()

async def listen():
    while listen_and_speak:
        response = ""
        with sr.Microphone() as source:
            print("Calibrating mic...")
            recognizer.adjust_for_ambient_noise(source)
            print("Listening...")
            audio = recognizer.listen(source)

            try:
                text = recognizer.recognize_google(audio)
                prompt = mp_handler.create_request()
                if not prompt:
                    await speak("I cannot see you right now.")
                    continue
                text += prompt
                response = await api.query(text)
                await speak(response)
            except sr.UnknownValueError:
                response = "Could not recognize speech."
                await speak(response)
            except sr.RequestError as e:
                response = f"Could not request results: {e}"
                await speak(response)
                print(response)

#edge_tts produces mp3, pyaudio needs pcm, so there's a conversion
async def speak(text):
    communicate = tts.Communicate(text, VOICE)

    mp3 = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3 += chunk["data"]
    audio_segment = AudioSegment.from_file(io.BytesIO(mp3), format="mp3")
    pcm = audio_segment.raw_data
    sample_width = audio_segment.sample_width
    sample_rate = audio_segment.frame_rate
    channels = audio_segment.channels

    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.get_format_from_width(sample_width), channels=channels, rate=sample_rate, output=True)
    stream.write(pcm)
    stream.stop_stream()
    stream.close()
    audio.terminate()

def main():
    asyncio.run(listen())