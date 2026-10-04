import os


UPLOAD_FOLDER = "uploads"  # folder where videos are stored

videos = []

for filename in os.listdir(UPLOAD_FOLDER):
    if filename.endswith(".webm"):
        # Extract username from filename
        username = filename.replace("_video.webm", "")
        # Full path to the video
        path = os.path.join(UPLOAD_FOLDER, filename)
        videos.append({"username": username, "path": path})

# Print the list
print(videos)