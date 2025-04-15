from flask import Flask, request, render_template, jsonify, send_from_directory
from local_tts import LocalTTS, Voice
import soundfile as sf
import numpy as np
import os
import uuid
import traceback

app = Flask(__name__)

# --- Configuration ---
# Ensure the 'static' directory exists for saving audio files
STATIC_DIR = 'static'
os.makedirs(STATIC_DIR, exist_ok=True)

AVAILABLE_VOICES = [
    'af_heart',
    'af_bella',
    'af_nicole',
    'af_sarah',
    'af_sky',
    'bf_emma',
    'bf_isabella',
    'am_adam',
    'am_michael',
    'bm_george',
    'bm_lewis',
] 

# --- Routes ---
@app.route('/')
def index():
    """Renders the main UI page."""
    return render_template('index.html', voices=AVAILABLE_VOICES)

@app.route('/generate', methods=['POST'])
def generate_audio():
    """Handles the TTS generation request."""
    try:
        data = request.json
        text = data.get('text', '')
        weights = data.get('weights', {})

        if not text:
            return jsonify({'error': 'Text input cannot be empty.'}), 400

        # Prepare voices list for LocalTTS, filtering out zero weights
        voices_to_blend: list[Voice] = []
        total_weight = 0
        for voice_id, weight in weights.items():
            if voice_id in AVAILABLE_VOICES and weight > 0:
                voices_to_blend.append({'voice_id': voice_id, 'weight': weight})
                total_weight += weight

        if not voices_to_blend:
            # Default to the first voice if none selected or all weights are zero
            default_voice = AVAILABLE_VOICES[0]
            voices_to_blend = [{'voice_id': default_voice, 'weight': 1.0}]
            print(f"No voices selected or all weights zero. Defaulting to: {default_voice}")
        elif total_weight > 0:
            # Normalize weights if they don't sum to 1 (optional, LocalTTS does this too)
            # You could also enforce sum=1 on the client-side
            for voice in voices_to_blend:
                voice['weight'] /= total_weight
            print(f"Using voices: {voices_to_blend}")
        else:
             # Handle case where weights were provided but all zero (should be caught by filter)
             return jsonify({'error': 'No voices selected with positive weight.'}), 400


        # Initialize TTS (Consider optimizing this - perhaps reuse instance?)
        # For simplicity, we re-initialize each time.
        # lang_code='a' allows mixing languages/voices. Adjust if needed.
        print("Initializing LocalTTS...")
        tts_instance = LocalTTS(lang_code='a', voices=voices_to_blend)
        if not tts_instance.pipeline or not tts_instance.pipeline.model:
             return jsonify({'error': 'Failed to initialize TTS model.'}), 500


        # Generate audio
        print(f"Generating audio for text: '{text[:50]}...'")
        audio_data = tts_instance.tts(text)

        if audio_data is None or audio_data.size == 0:
            return jsonify({'error': 'Failed to generate audio.'}), 500

        # Save audio to a unique file in the static directory
        filename = f"{uuid.uuid4()}.wav"
        filepath = os.path.join(STATIC_DIR, filename)
        print(f"Saving audio to: {filepath}")
        sf.write(filepath, audio_data, 24000) # Assuming 24kHz sample rate from Kokoro

        # Return the path to the audio file
        audio_url = f"/{STATIC_DIR}/{filename}"
        return jsonify({'audio_url': audio_url})

    except Exception as e:
        print(f"--- Error during audio generation ---")
        traceback.print_exc()
        print(f"--- End Error ---")
        return jsonify({'error': f'An internal error occurred: {str(e)}'}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serves static files (like the generated audio)."""
    return send_from_directory(STATIC_DIR, filename)

# --- Main Execution ---
if __name__ == '__main__':
    # Note: Use a proper WSGI server (like Gunicorn or Waitress) for production.
    # Set debug=False for production.
    app.run(debug=True, port=5001) # Use a different port if 5000 is common 