# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import soundfile as sf
import time
import uuid

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = MusicGen.get_pretrained('large', device='cuda')
        self.model.set_generation_params(duration=12)  # generate 8 seconds.
        # wav = model.generate_unconditional(4)    # generates 4 unconditional audio samples


        self.model_melody = MusicGen.get_pretrained('melody', device='cuda')
        self.model_melody.set_generation_params(duration=16)  # generate 8 seconds.
        # wav = model.generate_unconditional(4)    # generates 4 unconditional audio samples

    def predict(
        self,
        text: str = Input(description="Text prompt", default="Music to watch girls go by"),
        # scale: float = Input(
        #     description="Factor to scale image by", ge=0, le=10, default=1.5
        # ),
    ) -> Path:
        """Run a single prediction on the model"""
        
        start = time.time()
        wav = self.model.generate([text], progress=True)  # generates 3 samples.
        wav = wav[0]
        end = time.time()
        print(f"Generation took {end-start} seconds")

        save_path = f"/tmp/{uuid.uuid4()}"

        # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
        print(f"Saving {save_path}")
        audio_write(f'{save_path}', wav.cpu(), self.model.sample_rate, strategy="loudness")


        save_path = save_path + ".wav"
        # return path to wav file
        return Path(save_path)
    

