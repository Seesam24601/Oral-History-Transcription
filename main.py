import whisper
import os


# Get the names of all of the files in the data folder
files = [f for f in os.listdir("data") if os.path.isfile(os.path.join("data", f))]

# Use tiny model for now to test quickly
model = whisper.load_model("base")


# Transcribe each file
for file in files:

    # fp16 is only support for GPU encoding
    result = model.transcribe("data\\" + file, fp16 = False)

    # Save to .txt file in output folder
    output_file_name = "output\\" + file.partition(".")[0] + ".txt"
    with open(output_file_name, "w") as output_file:
        output_file.write(result["text"])
        