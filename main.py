import os
import whisper
from simple_diarizer.diarizer import Diarizer
import json
import numpy as np

model = whisper.load_model("medium")
diarization = Diarizer(embed_model='xvec', cluster_method='sc')

def convertJson(file: str):
    sentences = model.transcribe("audio/" + file, word_timestamps=True)["segments"]
    segments = diarization.diarize("audio/" + file)
    jsonArray = []

    for sentence in sentences:
        end = float(sentence['words'][-1]['end'])
        if len(segments) == 0: break;
        currentIndex: int
        diff = abs(float(segments[0]['end']) - end)
        for i in range(1, len(segments)):
            currentIndex = i
            newDiff = abs(float(segments[i]['end']) - end)
            if newDiff > diff: break
        first = convert_np_floats(sentence["words"])
        first = convert_np_ints(first)
        first = convert_np_ints32(first)

        second = convert_np_floats(segments[:currentIndex])
        second = convert_np_ints(second)
        second = convert_np_ints32(second)

        jsonArray.append((first, second))
        segments = segments[currentIndex:]
    
    json_text = json.dumps(jsonArray, indent=4)
    with open(f"output/{file.split('.')[0]}.json", 'w') as f:
        f.write(json_text)

def convert_np_floats(obj):
    if isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_np_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_floats(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_np_floats(x) for x in obj)
    else:
        return obj

def convert_np_ints(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_np_ints(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_ints(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_np_ints(x) for x in obj)
    else:
        return obj

def convert_np_ints32(obj):
    if isinstance(obj, np.int32):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_np_ints32(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_ints32(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_np_ints32(x) for x in obj)
    else:
        return obj
def main():
    files = os.listdir("audio/")
    for file in files:
        if file == ".gitignore":
            continue
        convertJson(file)
    postFiles = os.listdir("audio/")
    for file in postFiles:
        if not file in files:
            os.remove(f"audio/{file}")

if __name__ == "__main__":
    main()
