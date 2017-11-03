import subprocess
import os


source_directory = "../../dataset/"
target_directory = "../../audio_dataset/"
directory_path = os.fsencode(source_directory)
for file in os.listdir(directory_path):
    filename = os.fsdecode(file)
    if filename.endswith(".mp4"):
        full_filename = os.path.abspath(source_directory) + "\\" + filename
        target_filename = os.path.abspath(target_directory) + "\\" + filename + ".wav"
        print("Convert: " + full_filename)
        command = "ffmpeg -i " + full_filename + " -ab 160k -ac 2 -ar 44100 -vn " + target_filename
        subprocess.call(command, shell=True)


