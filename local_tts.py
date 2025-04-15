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
Households in our base specification are composed of a female ( f ) and a male (m) of equal age.
The model periods are in months indexed by t = 1, 2, . . . , Tmax, where Tmax is the month of death of
the last remaining survivor from the couple. Each member of the household is eligible to work and
save starting from the first month of age 25 (t = 1). The retirement date is denoted Tret, and our base
case specifies an exogenous retirement age of 65, such that Tret = 480. An individual may experience
nonemployment during their potential working years, such that not all investors work the full 40 years.
In the base case, we assume that individuals save rc = 10% of their labor income for retirement, and
no contributions occur during nonemployment periods. The assumed 10% contribution rate is close
to the mean (11.7%) and median (11.0%) contribution rates for participants in Vanguard definedcontribution plans in 2023, including both employee and employer contributions [Vanguard (2024)].
16
We also assume that individuals earning less than Ymin = $15,000 (in 2022 USD) in a given year forego
contributing to their retirement plan, consistent with evidence of low retirement saving rates among
this group [e.g., Vanguard (2024)].
At time Tret+1, each individual leaves the workforce (ending either employment or nonemployment)
and begins to draw from retirement savings and Social Security. We assume that investors withdraw
rw = 4% of their account balance at retirement in the first year and inflation-adjusted amounts calculated from this base withdrawal in subsequent years [i.e., the “4% rule” of Bengen (1994)]. In reality,
retirees use a variety of withdrawal strategies. The 4% rule is ubiquitous in popular press and common
retirement advice, so we use it as a simple heuristic for retirement withdrawals.17 We also demonstrate that our main conclusions hold for alternative retirement withdrawal rules. We note that the outcomes
of households who choose to annuitize fully at retirement will be reflected by our wealth at retirement
results.
The Social Security Administration (SSA) reports conditional death probabilities at each age for
females and males.18 Our simulations incorporate gender-specific longevity risk, and the lifespan of
each individual is randomly determined. Both the female and the male in each couple are alive at
age 25, but one or both may die before retirement at age 65. There is considerable uncertainty over
longevity outcomes. The 5th percentile of age at death for the couple (i.e., the last survivor) is 70.8
years, and the 95th percentile is 100.0 years. This uncertainty is an important feature to consider in
assessing the ability of investment strategies to fund consumption through retirement (see the internet
appendix for further details on the distribution of age at death).
Given the simulation design, the (unmodeled) consumption and potential survivor benefits from
Social Security during the pre-retirement period are independent of the retirement investment strategy.
As such, we do not study consumption in the pre-retirement period and do not include it in the utility
calculations.
    '''

    # [{'voice_id': 'af_heart', 'weight': 0.5}, {'voice_id': 'am_michael', 'weight': 1}]
    local_tts = LocalTTS(lang_code='a', voices=[{'voice_id': 'af_heart', 'weight': 1}])
    local_tts.save_audio(text, 'Lifecycle design.wav')

