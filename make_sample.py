"""生成一段离线语音样例 sample.wav（使用 Windows 自带 TTS），用于本地测试转录。"""
import pyttsx3

TEXT = (
    "Hello, this is a local test of the speech to text system. "
    "We are converting audio into a markdown document entirely on this machine. "
    "The quick brown fox jumps over the lazy dog."
)

engine = pyttsx3.init()
engine.setProperty("rate", 160)
engine.save_to_file(TEXT, "sample.wav")
engine.runAndWait()
print("wrote sample.wav")
print("text:", TEXT)
