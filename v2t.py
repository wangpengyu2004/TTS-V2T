import whisper
def speech_to_text(file_path):
    # 加载 Whisper 模型
    model = whisper.load_model("medium")  # 你可以选择其他模型，如 "small", "medium", "large"
    
    # 加载音频文件
    audio = whisper.load_audio(file_path)
    audio = whisper.pad_or_trim(audio)
    # 转换语音为文本
    result = model.transcribe(audio)
    # 打印识别结果
    print(f"result: {result['text']}")

if __name__ == "__main__":
    file_path = "/home/wpy/wpy_workspace/s2s/VITS/raw_audio/wpy_1.wav"  # 替换成你音频文件的路径
    speech_to_text(file_path)