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


def basic():

    pipeline = KPipeline(lang_code='a')
    text = '''
    [Kokoro](/kˈOkəɹO/) is an open-weight TTS model with 82 million parameters. Despite its lightweight architecture, it delivers comparable quality to larger models while being significantly faster and more cost-efficient.
    '''
    voice1 = 'af_heart'
    voice2 = 'am_michael'
    generator1 = pipeline(text, voice=voice1)
    generator2 = pipeline(text, voice=voice2)

    # Adjustable weights for mixing
    weight1 = 0.75  # 75% volume for voice1
    weight2 = 1.0   # 100% volume for voice2

    # Process generator1 and save individual audio
    for i, (gs, ps, audio1) in enumerate(generator1):
        print(f"{voice1} - chunk {i}, gs: {gs}, ps: {ps}")
        sf.write(f'{voice1}.wav', audio1, 24000)

def blend_voices_custom_weights(text: str):

    """
    Blends voice embeddings with custom weights using KPipeline's
    voice loading mechanism.
    """
    # Initialize the pipeline
    # This will also initialize the model (KModel) by default
    # and determine the device (e.g., 'cuda' or 'cpu')
    pipeline = KPipeline(lang_code='a') # Or your desired language
    if not pipeline.model:
        print("Error: Pipeline initialized without a model. Cannot generate audio.")
        return
    device = pipeline.model.device
    print(f"Pipeline model initialized on device: {device}")

    # --- Define voices and custom weights ---
    voice1_id = 'af_heart'
    voice2_id = 'am_michael'
    weight1 = 0.5  
    weight2 = 1 

    print(f"Blending {voice1_id} ({weight1*100}%) and {voice2_id} ({weight2*100}%)")

    try:
        # --- Load individual voice tensors using the pipeline ---
        # load_single_voice will download from HF Hub if needed and cache it.
        # It returns the tensor, likely on CPU initially.
        print(f"Loading voice tensor for: {voice1_id}")
        tensor1 = pipeline.load_single_voice(voice1_id)
        print(f"Loading voice tensor for: {voice2_id}")
        tensor2 = pipeline.load_single_voice(voice2_id)

        # --- Perform custom weighted blending ---
        # Move tensors to the model's device before blending
        tensor1 = tensor1.to(device)
        tensor2 = tensor2.to(device)

        # Ensure weights sum to 1 for a proper weighted average
        total_weight = weight1 + weight2
        blended_voice_tensor = (weight1 * tensor1 + weight2 * tensor2) / total_weight
        print("Created blended voice tensor.")

        # --- Generate audio using the blended tensor ---
        # Pass the *tensor* directly as the 'voice' argument
        print("Generating audio with the blended tensor...")
        blended_generator = pipeline(text, voice=blended_voice_tensor)

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
            # output_filename = f"blended_custom_{voice1_id}_{weight1:.1f}_{voice2_id}_{weight2:.1f}.wav"
            # sf.write(output_filename, final_blended_audio, 24000)
            # print(f"Created {output_filename}")

            return final_blended_audio
        else:
            print("No audio generated for the blended voice.")

    except FileNotFoundError as e:
         print(f"\nError: Could not find a voice file. {e}")
         print("Please ensure the voice IDs exist in the Hugging Face repository")
         print(f"associated with the pipeline (default: {pipeline.repo_id})")
         print("and that you have an internet connection for the first download.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        traceback.print_exc()

    # --- Optional: Generate originals for comparison ---
    # Comment out for now, useful for testing
    # print(f"\nGenerating original voice: {voice1_id}")
    # original_gen1 = pipeline(text, voice=voice1_id)
    # original_audio1_chunks = [res.audio.cpu().numpy() for res in original_gen1 if res.audio is not None]
    # if original_audio1_chunks:
    #     sf.write(f'original_{voice1_id}.wav', np.concatenate(original_audio1_chunks), 24000)

    # print(f"Generating original voice: {voice2_id}")
    # original_gen2 = pipeline(text, voice=voice2_id)
    # original_audio2_chunks = [res.audio.cpu().numpy() for res in original_gen2 if res.audio is not None]
    # if original_audio2_chunks:
    #     sf.write(f'original_{voice2_id}.wav', np.concatenate(original_audio2_chunks), 24000)


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
    


if __name__ == "__main__":
    text = '''
    [Kokoro](/kˈOkəɹO/) is an open-weight TTS model with 82 million parameters. Despite its lightweight architecture, it delivers comparable quality to larger models while being significantly faster and more cost-efficient.
    '''
    blend_voices_custom_weights(text)

