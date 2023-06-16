import subprocess
import os
import uuid
import shutil
import glob
import json
import random
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import re
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
# time the generation
import time


model = MusicGen.get_pretrained('large', device='cuda')
# generate 8 seconds.
# wav = model.generate_unconditional(4)    # generates 4 unconditional audio samples


model_melody = MusicGen.get_pretrained('melody', device='cuda')
model_melody.set_generation_params(duration=16)  # generate 8 seconds.
# wav = model.generate_unconditional(4)    # generates 4 unconditional audio samples

app = FastAPI()

# endpoint that generates texst from audio /musicgen/{prompt}?durat
@app.get("/musicgen/{text:path}")
def musicgen(text: str = "", duration: int = 12,response_class=FileResponse):

    model.set_generation_params(duration=min(duration, 30))  

    start = time.time()
    wav = model.generate([text], progress=True)  # generates 3 samples.
    wav = wav[0]
    end = time.time()
    print(f"Generation took {end-start} seconds")

    save_path = f"/tmp/{uuid.uuid4()}"

    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    print(f"Saving {save_path}")
    audio_write(f'{save_path}', wav.cpu(), model.sample_rate, strategy="loudness")

    save_path = save_path + ".wav"

    # shutil.rmtree(save_path)

    return FileResponse(save_path, media_type="audio/wav")


# endpoint that allows uploading an audio file. file should be optional
@app.post("/musicgen_transfer/{text}")
def musicgen_transfer(text: str = "", file: UploadFile = File(None), response_class=FileResponse):
    save_path = f"/tmp/{uuid.uuid4()}"

    # create save path
    os.makedirs(save_path, exist_ok=True)

    # save file
    if file:
        print("got file. writing to disk at", f"{save_path}/input.wav")
        with open(f"{save_path}/input.wav", "wb") as f:
            f.write(file.file.read())
    
    melody, sr = torchaudio.load(f"{save_path}/input.wav")
    # generates using the melody from the given audio and the provided descriptions.
    wav = model_melody.generate_with_chroma([text], melody[None].expand(1, -1, -1), sr, progress=True)[0]

    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    print(f"Saving {save_path}/output.wav")
    audio_write(f'{save_path}/output', wav.cpu(), model_melody.sample_rate, strategy="loudness")

    # shutil.rmtree(save_path)

    return FileResponse(f'{save_path}/output.wav', media_type="audio/wav")


# curl request
# curl -X POST "http://localhost:8000/audioldm_transfer/this%20is%20a%20test" -H "accept: audio/wav" -H "Content-Type: multipart/form-data" -F "file=@trumpet.wav"

# listen

# uvicorn musicgen_server:app --reload 

#  ffmpeg -r 6 -i outputs/txt2img-images/2023-05-12/%*.png -r 6 nicolaout.mp4