import os
import sys
import datetime
import time
import openai
import dotenv
import streamlit as st
import pyaudio
import wave
import threading
import queue
import numpy as np
from scipy.io import wavfile
import noisereduce as nr
import warnings

# import API key from .env file
dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

#keeping GPT response simple, mi casa su casa :)
system_prompt = "You are a helpful assistant who answers in short 1 liners."

CHUNK = 1024
CHANNELS = 1  # Set to 1 or 2 based on your device
FORMAT = pyaudio.paInt16
RATE = 44100
MIN_RECORD_SECONDS = 1
MAX_RECORD_SECONDS = 30
SILENCE_THRESHOLD = 0.01
SILENCE_DURATION = 1.0  # seconds of silence to stop recording

def transcribe(file_path):
    with open(file_path, "rb") as audio_file:
        transcript = openai.audio.transcriptions.create(model="whisper-1", file=audio_file)
        print(transcript.text)
    return transcript.text

def generate_gpt_response(transcript):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript}
        ]
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content

# sdf

def is_silent(data_chunk, threshold):
    """Check if the audio chunk is silent."""
    return np.max(np.abs(np.frombuffer(data_chunk, dtype=np.int16))) < threshold * 32767

def record_audio_continuous(device_index, audio_queue, stop_event):
    p = pyaudio.PyAudio()
    
    try:
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        input_device_index=device_index,
                        frames_per_buffer=CHUNK)

        print(f"Recording started with device index: {device_index}, Channels: {CHANNELS}, Rate: {RATE}")

        while not stop_event.is_set():
            data = stream.read(CHUNK)
            audio_queue.put(data)

        stream.stop_stream()
        stream.close()

    except Exception as e:
        print(f"Error recording audio: {str(e)}")
    finally:
        p.terminate()

def process_audio(audio_queue, message_queue, stop_event):
    silence_counter = 0
    recording = []
    is_recording = False
    start_time = None

    while not stop_event.is_set():
        try:
            chunk = audio_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        if not is_recording and not is_silent(chunk, SILENCE_THRESHOLD):
            is_recording = True
            recording = [chunk]
            start_time = time.time()
            print("Recording started")
        elif is_recording:
            recording.append(chunk)

            if is_silent(chunk, SILENCE_THRESHOLD):
                silence_counter += 1
            else:
                silence_counter = 0

            duration = time.time() - start_time
            if silence_counter > int(SILENCE_DURATION * RATE / CHUNK) or duration > MAX_RECORD_SECONDS:
                if duration > MIN_RECORD_SECONDS:
                    print(f"Processing audio of length: {duration:.2f} seconds")
                    file_path = save_audio_file(recording)
                    print(f"Audio saved to {file_path}")
                    
                    message_queue.put({"user": f"Audio processed: {file_path}", "assistant": f"Audio duration: {duration:.2f} seconds"})

                is_recording = False
                recording = []
                silence_counter = 0
                start_time = None
                print("Finished processing audio chunk")

def save_audio_file(audio_data):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"audio_{timestamp}.wav"

    with wave.open(file_name, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pyaudio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(audio_data))

    # Verify the saved file
    with wave.open(file_name, 'rb') as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        duration = frames / float(rate)
        print(f"Saved audio file details: channels={wf.getnchannels()}, frames={frames}, "
              f"sample_width={wf.getsampwidth()}, frame_rate={rate}")
        print(f"Audio duration: {duration:.2f} seconds")

    return file_name

def get_audio_devices():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    num_devices = info.get('deviceCount')
    
    devices = []
    for i in range(num_devices):
        device_info = p.get_device_info_by_host_api_device_index(0, i)
        if device_info.get('maxInputChannels') > 0:
            devices.append((i, device_info.get('name')))
    
    p.terminate()
    return devices

def is_silent(data_chunk, threshold):
    """Check if the audio chunk is silent."""
    return np.max(np.abs(np.frombuffer(data_chunk, dtype=np.int16))) < threshold

def record_audio_continuous(device_index, audio_queue, stop_event):
    p = pyaudio.PyAudio()
    device_info = p.get_device_info_by_index(device_index)
    channels = min(2, device_info.get('maxInputChannels'))

    def callback(in_data, frame_count, time_info, status):
        if not stop_event.is_set():
            audio_queue.put(in_data)
        return (None, pyaudio.paContinue)

    try:
        stream = p.open(format=FORMAT,
                        channels=channels,
                        rate=RATE,
                        input=True,
                        input_device_index=device_index,
                        frames_per_buffer=CHUNK,
                        stream_callback=callback)

        print(f"Recording started with device: {device_info.get('name')}, Channels: {channels}, Rate: {RATE}")
        stream.start_stream()

        while not stop_event.is_set():
            time.sleep(0.1)

        stream.stop_stream()
        stream.close()

    except Exception as e:
        print(f"Error recording audio: {str(e)}")
    finally:
        p.terminate()

def process_audio(audio_queue, message_queue, stop_event):
    silence_counter = 0
    recording = []
    is_recording = False

    while not stop_event.is_set():
        try:
            chunk = audio_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        if not is_recording and not is_silent(chunk, SILENCE_THRESHOLD * 32767):
            is_recording = True
            recording = [chunk]
            print("Recording started")
        elif is_recording:
            recording.append(chunk)

            if is_silent(chunk, SILENCE_THRESHOLD * 32767):
                silence_counter += 1
            else:
                silence_counter = 0

            if silence_counter > int(SILENCE_DURATION * RATE / CHUNK) or len(recording) > MAX_RECORD_SECONDS * RATE / CHUNK:
                if len(recording) > MIN_RECORD_SECONDS * RATE / CHUNK:
                    print(f"Processing audio of length: {len(recording) * CHUNK / RATE:.2f} seconds")
                    try:
                        file_path = save_audio_file(recording)
                        print(f"Audio saved to {file_path}")
                        
                        # Apply noise reduction
                        rate, data = wavfile.read(file_path)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", RuntimeWarning)
                            try:
                                reduced_noise = nr.reduce_noise(y=data, sr=rate)
                                wavfile.write(file_path, rate, reduced_noise.astype(np.int16))
                                print("Noise reduction applied")
                            except Exception as e:
                                print(f"Error during noise reduction: {str(e)}")
                                reduced_noise = data
                        
                        transcript_text = transcribe(file_path)
                        print(f"Whisper Transcript: {transcript_text}")
                        
                        if transcript_text.strip():
                            gpt_response = generate_gpt_response(transcript_text)
                            print(f"GPT Response: {gpt_response}")
                            message_queue.put({"user": transcript_text, "assistant": gpt_response})

                        os.remove(file_path)
                    except Exception as e:
                        print(f"Error processing audio: {str(e)}")

                is_recording = False
                recording = []
                silence_counter = 0
                print("Recording stopped")

def main():
    st.title("Continuous Whisper Transcription with Flexible Audio Input")

    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'recording' not in st.session_state:
        st.session_state.recording = False
    if 'message_queue' not in st.session_state:
        st.session_state.message_queue = queue.Queue()

    devices = get_audio_devices()
    device_names = [f"{name} (ID: {id})" for id, name in devices]

    col1, col2, col3 = st.columns(3)
    selected_device = col1.selectbox("Select Audio Source", device_names)
    
    if not st.session_state.recording:
        if col2.button("Start Recording"):
            st.session_state.recording = True
            st.session_state.start_time = time.time()
            device_index = devices[device_names.index(selected_device)][0]
            
            audio_queue = queue.Queue()
            stop_event = threading.Event()
            
            recording_thread = threading.Thread(target=record_audio_continuous, args=(device_index, audio_queue, stop_event))
            processing_thread = threading.Thread(target=process_audio, args=(audio_queue, st.session_state.message_queue, stop_event))
            
            recording_thread.start()
            processing_thread.start()

            st.session_state.stop_event = stop_event
            st.session_state.threads = (recording_thread, processing_thread)
            st.session_state.audio_queue = audio_queue
    else:
        if col2.button("Stop Recording"):
            st.session_state.recording = False
            if hasattr(st.session_state, 'stop_event'):
                st.session_state.stop_event.set()
                for thread in st.session_state.threads:
                    thread.join()
            
            if hasattr(st.session_state, 'audio_queue'):
                while not st.session_state.audio_queue.empty():
                    try:
                        st.session_state.audio_queue.get_nowait()
                    except queue.Empty:
                        break

    if col3.button("Clear Chat"):
        st.session_state.messages = []

    chat_container = st.container()


        # Process any new messages from the queue
    new_messages = False
    while not st.session_state.message_queue.empty():
        try:
            message = st.session_state.message_queue.get_nowait()
            st.session_state.messages.append({"role": "user", "content": message["user"]})
            st.session_state.messages.append({"role": "assistant", "content": message["assistant"]})
            new_messages = True
        except queue.Empty:
            break

    # Display chat messages
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    if st.session_state.messages:
        chat_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
        st.download_button("Download Chat", chat_text)

    if st.session_state.recording:
        st.write("Recording in progress...")
    else:
        st.write("Not recording. Press 'Start Recording' to begin.")

    # Update the app state
    if st.session_state.recording or new_messages:
        st.rerun()

if __name__ == "__main__":
    working_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(working_dir)
    main()
