from llama_index.readers.youtube_transcript import YoutubeTranscriptReader

loader = YoutubeTranscriptReader()
documents = loader.load_data(
    ytlinks=["https://www.youtube.com/watch?v=i3OYlaoj-BM"]
)

print(documents)