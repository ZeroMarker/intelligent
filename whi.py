import whisper

# Audio file path
PATH = "../../../dataset/gettysburg10.wav"

# load model && transcribe
model = whisper.load_model("base")
result = model.transcribe(PATH, fp16=False)

# Output
print(result["text"])
