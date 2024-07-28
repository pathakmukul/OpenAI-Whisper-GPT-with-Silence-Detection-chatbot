# OpenAI-Whisper-GPT-with-Silence-Detection-chatbot
Real-time audio recording and transcription system using Streamlit, and OpenAI's Whisper model on Streamlit. With continuous audio capture with silence detection
![DALL·E 2024-07-28 14 37 25 - Create a radical, eye-catching graphic for a tech project titled 'OpenAI-GPT-Whisper-with-Silence-Detection-chatbot'  The graphic should feature dynam copy](https://github.com/user-attachments/assets/3ebba866-c947-4f96-a21a-d11109b3ea81)


## Overview

This project implements a real-time audio recording and transcription system using Python, Streamlit, and OpenAI's Whisper model. It features continuous audio capture with silence detection, allowing for automatic segmentation of audio input. The system is designed to work with various audio input devices and can be easily integrated into applications requiring speech-to-text functionality.

## Features

- Real-time audio recording from selected input devices
- Silence detection for automatic audio segmentation
- Integration with OpenAI's Whisper model for transcription
- Simple and interactive Streamlit-based user interface
- Continuous processing and transcription of audio segments
- Configurable parameters for silence detection and recording duration

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.7 or higher
- pip (Python package manager)
- An OpenAI API key for using the Whisper & GPT model

## Installation

1. Clone the repository:
   ``` 
   git clone https://github.com/pathakmukul/OpenAI-Whisper-GPT-with-Silence-Detection-chatbot.git
   cd continuous-audio-transcription
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up your OpenAI API key:
   - Create a `.env` file in the project root directory
   - Add your API key to the file:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```

## Usage

To run the application:

1. Ensure your virtual environment is activated (if you're using one).

2. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

3. The application will open in your default web browser.

4. Select your audio input device from the dropdown menu.

5. Click "Start Recording" to begin capturing audio.

6. Speak into your microphone. The application will automatically detect silence and process audio segments.

7. Click "Stop Recording" when you're finished.

8. View the transcribed text and any assistant responses in the chat interface.

## Configuration

You can adjust the following parameters in the `app.py` file to fine-tune the audio processing:

- `CHUNK`: Size of audio chunks (default: 1024)
- `RATE`: Audio sample rate (default: 44100)
- `MIN_RECORD_SECONDS`: Minimum duration of an audio segment (default: 1 second)
- `MAX_RECORD_SECONDS`: Maximum duration of an audio segment (default: 30 seconds)
- `SILENCE_THRESHOLD`: Threshold for silence detection (default: 0.01)
- `SILENCE_DURATION`: Duration of silence to trigger processing (default: 1.0 second)

## Troubleshooting

If you encounter issues with audio recording or processing:

1. Check your microphone settings and ensure it's properly connected.
2. Verify that you've selected the correct audio input device in the application.
3. Check the console output for any error messages or warnings.
4. Ensure your OpenAI API key is correctly set in the `.env` file.

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- OpenAI for the Whisper model
- Streamlit for the web application framework
- PyAudio for audio processing capabilities

## Contact

If you have any questions or feedback, please contact Mukul Pathak at tpmmukul@gmail.com
