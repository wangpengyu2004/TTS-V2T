import os
import numpy as np
import torch
from torch import no_grad, LongTensor
import argparse
import commons
from mel_processing import spectrogram_torch
import utils
from models import SynthesizerTrn
import librosa
import logging
import soundfile as sf
from text import text_to_sequence

device = "cuda:0" if torch.cuda.is_available() else "cpu"
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)

language_marks = {
    "Japanese": "",
    "日本語": "[JA]",
    "简体中文": "[ZH]",
    "English": "[EN]",
    "Mix": "",
}
lang = ['日本語', '简体中文', 'English', 'Mix']

def get_text(text, hps, is_symbol):
    text_norm = text_to_sequence(text, hps.symbols, [] if is_symbol else hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm

def create_tts_fn(model, hps, speaker_ids):
    def tts_fn(text, speaker, language, speed):
        if language is not None:
            text = language_marks[language] + text + language_marks[language]
        speaker_id = speaker_ids[speaker]
        stn_tst = get_text(text, hps, False)
        with no_grad():
            x_tst = stn_tst.unsqueeze(0).to(device)
            x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)
            sid = LongTensor([speaker_id]).to(device)
            audio = model.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8,
                                length_scale=1.0 / speed)[0][0, 0].data.cpu().float().numpy()
        del stn_tst, x_tst, x_tst_lengths, sid
        return audio

    return tts_fn

def create_vc_fn(model, hps, speaker_ids):
    def vc_fn(original_speaker, target_speaker, input_audio_path):
        if input_audio_path is None:
            return "You need to provide an audio file", None
        audio, sampling_rate = librosa.load(input_audio_path, sr=hps.data.sampling_rate)
        original_speaker_id = speaker_ids[original_speaker]
        target_speaker_id = speaker_ids[target_speaker]

        audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.transpose(1, 0))
        if sampling_rate != hps.data.sampling_rate:
            audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=hps.data.sampling_rate)
        with no_grad():
            y = torch.FloatTensor(audio)
            y = y / max(-y.min(), y.max()) / 0.99
            y = y.to(device)
            y = y.unsqueeze(0)
            spec = spectrogram_torch(y, hps.data.filter_length,
                                     hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                                     center=False).to(device)
            spec_lengths = LongTensor([spec.size(-1)]).to(device)
            sid_src = LongTensor([original_speaker_id]).to(device)
            sid_tgt = LongTensor([target_speaker_id]).to(device)
            audio = model.voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt)[0][
                0, 0].data.cpu().float().numpy()
        del y, spec, spec_lengths, sid_src, sid_tgt
        return audio

    return vc_fn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="./OUTPUT_MODEL/G_latest.pth", help="directory to your fine-tuned model")
    parser.add_argument("--config_dir", default="./OUTPUT_MODEL/config.json", help="directory to your model config file")
    parser.add_argument("--input_text", default=None, help="input text for TTS")
    parser.add_argument("--input_audio", default=None, help="input audio file for VC")
    parser.add_argument("--output_audio", default="./output.wav", help="output audio file path")
    parser.add_argument("--speaker", default="speaker1", help="speaker name for TTS/VC")
    parser.add_argument("--language", default="简体中文", help="language for TTS")
    parser.add_argument("--speed", type=float, default=1.0, help="speed for TTS")

    args = parser.parse_args()
    hps = utils.get_hparams_from_file(args.config_dir)

    net_g = SynthesizerTrn(
        len(hps.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).to(device)
    _ = net_g.eval()

    _ = utils.load_checkpoint(args.model_dir, net_g, None)
    speaker_ids = hps.speakers
    speakers = list(hps.speakers.keys())
    tts_fn = create_tts_fn(net_g, hps, speaker_ids)
    vc_fn = create_vc_fn(net_g, hps, speaker_ids)

  # 处理TTS如果输入文本提供
    if args.input_text is not None:
        audio = tts_fn(args.input_text, args.speaker, args.language, args.speed)
        sf.write(args.output_audio, audio, hps.data.sampling_rate)
        print(f"TTS音频已保存至 {args.output_audio}")

    # 处理VC如果输入音频提供
    if args.input_audio is not None:
        audio = vc_fn(args.speaker, args.speaker, args.input_audio)
        sf.write(args.output_audio, audio, hps.data.sampling_rate)
        print(f"转换后的音频已保存至 {args.output_audio}")
