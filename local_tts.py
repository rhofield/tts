'''
I tried several methods to blend voices, but they all had issues.

1. Using onnx models didn't work as a I couldnt access the tensors as the demo I followed used: from models import build_model, which didn't work 
2. Overlapping speach doesn't work, it's just 2 audio files ontop of each other, not the intdended result
3. Onnx models also sounded worse and phonems weren't as good
4. Using the built in method with kokoro worked but you could only do a 50/50 blend
'''
from kokoro import KPipeline
import soundfile as sf
import numpy as np
import traceback
from typing import TypedDict

class Voice(TypedDict):
    voice_id: str
    weight: float

class LocalTTS:
    def __init__(self, lang_code: str = 'a', voices: list[Voice] = [{'voice_id': 'af_heart', 'weight': 1}]):
        self.pipeline = KPipeline(lang_code=lang_code)
        if not self.pipeline.model:
            print("Error: Pipeline initialized without a model. Cannot generate audio.")
            return
        
        self.device = self.pipeline.model.device
        self.voice_tensor = self.get_voice_tensor(voices)

    def get_voice_tensor(self, voices: list[Voice] ):
        try:
            # --- Load individual voice tensors using the pipeline ---
            # load_single_voice will download from HF Hub if needed and cache it.
            # It returns the tensor, likely on CPU initially.
            tensors = []
            for voice in voices:
                print(f"Loading voice tensor for: {voice['voice_id']}")
                tensor = self.pipeline.load_single_voice(voice['voice_id'])

                # --- Perform custom weighted blending ---
                # Move tensors to the model's device before blending
                tensor = tensor.to(self.device)
                tensors.append(tensor)
            
            # Ensure weights sum to 1 for a proper weighted average
            total_weight = sum(voice['weight'] for voice in voices)
            blended_voice_tensor = sum(voice['weight'] * tensor for voice, tensor in zip(voices, tensors)) / total_weight

            return blended_voice_tensor

        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            traceback.print_exc()
        
    
    def tts(self, text: str):
        """
        Uses a custom voice tensor to generate audio.
        """
        # --- Generate audio using the blended tensor ---
        # Pass the *tensor* directly as the 'voice' argument
        print("Generating audio with the blended tensor...")
        blended_generator = self.pipeline(text, voice=self.voice_tensor)

        # Process the generated audio chunks
        blended_audio_chunks = []
        for result in blended_generator:
            print(f"Generated chunk for blended voice (Graphemes: {result.graphemes[:20]}...)")
            if result.audio is not None:
                # Audio is a torch tensor, move to CPU and convert to numpy
                blended_audio_chunks.append(result.audio.cpu().numpy())
            else:
                print(f"Warning: Chunk had no audio.")

        # Concatenate and save the final audio
        if blended_audio_chunks:
            final_blended_audio = np.concatenate(blended_audio_chunks)
            return final_blended_audio
        else:
            print("No audio generated for the blended voice.")
    
    def save_audio(self, text: str, filename: str):
        audio = self.tts(text)
        sf.write(filename, audio, 24000)


if __name__ == "__main__":
    text = '''
    [Kokoro](/kˈOkəɹO/) is an open-weight TTS model with 82 million parameters. Despite its lightweight architecture, it delivers comparable quality to larger models while being significantly faster and more cost-efficient.
    '''
    local_tts = LocalTTS(lang_code='a', voices=[{'voice_id': 'af_heart', 'weight': 0.5}, {'voice_id': 'am_michael', 'weight': 1}])
    local_tts.save_audio(text, 'test.wav')

