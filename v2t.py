import whisper
import argparse
def speech_to_text(args):
    # 加载 Whisper 模型
    model = whisper.load_model(args.size)  # 你可以选择其他模型，如 "small", "medium", "large"
    # 加载音频文件
    audio = whisper.load_audio(args.file_path)
    audio = whisper.pad_or_trim(audio)
    # 转换语音为文本
    result = model.transcribe(audio)
    # 打印识别结果
    print(f"result: {result['text']}")
    with open(args.output_path,"w") as f:
        f.write(result)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", default="./file_path.wav", help="your file path")
    parser.add_argument("--size", default="small", help=" your model size")
    parser.add_argument("--output_path", default="./output_path.txt", help="your file path")
    args = parser.parse_args()
    speech_to_text(args)
