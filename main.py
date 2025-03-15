
import onnxruntime
from f0Analyzer import F0Analyzer,resample_align_curve
from wav2mel import PitchAdjustableMelSpectrogram
import dataclasses
import soundfile as sf
import numpy as np
import torch

@dataclasses.dataclass
class Config:
    sample_rate: int = 44100
    win_size: int = 2048
    hop_size: int = 512
    n_mels: int = 128
    n_fft: int = 2048
    mel_fmin: float = 40
    mel_fmax: float = 16000.0
    f0_extractor: str = 'parselmouth'
    f0_min: float = 65
    f0_max: float = 1600
    vocoder_path: str = r"path/to/vocoder/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.onnx" #必须使用pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.onnx模型
def dynamic_range_compression_torch(x, C=1, clip_val=1e-9):
    return torch.log(torch.clamp(x, min=clip_val) * C)
def wave_to_mel(wave,mel_keyshift,speed):
    '''
    wave shape=(n_samples,)
    mel_keyshift: float
    speed: float,                        不变速为1.0
    '''
    wave = torch.from_numpy(wave).float()
    melAnalysis = PitchAdjustableMelSpectrogram(
        sample_rate=Config.sample_rate, 
        n_fft=Config.n_fft, 
        win_length=Config.win_size, 
        hop_length=Config.hop_size, 
        f_min=Config.mel_fmin, 
        f_max=Config.mel_fmax,
        n_mels=Config.n_mels
        )
    mel = melAnalysis(
        wave.unsqueeze(0),
        mel_keyshift,
        speed,
        ).squeeze()
    
    mel = dynamic_range_compression_torch(mel)
    
    f0Analyzer = F0Analyzer(
        sampling_rate = Config.sample_rate,
        f0_extractor = Config.f0_extractor,
        hop_size = Config.hop_size,
        f0_min = Config.f0_min,
        f0_max = Config.f0_max
        )
    
    f0, _ = f0Analyzer(wave, n_frames=mel.shape[1],speed=speed)
    
    #根据mel_keyshift调整f0
    f0 = f0*(2 ** (mel_keyshift / 12))

    # mel shape: (n_mels, n_frames*speed)
    # f0 shape:  (n_frames*speed, )

    return mel, f0

def mel_to_wave(mel,f0,vocoder_keyshift,speed):
    '''
    mel shape = (n_mels, n_frames*speed)
    f0 shape = (n_frames*speed,)
    vocoder_keyshift shape = (n_frames,)
    '''
    timestep=Config.hop_size/Config.sample_rate
    vocoder_keyshift = resample_align_curve(
        vocoder_keyshift, 
        timestep, 
        timestep*speed, 
        mel.shape[1]
    ) #调整vocoder_keyshift到mel的长度
    f0 = f0*(2 ** (vocoder_keyshift / 12))

    # 加载vocoder模型
    ort_session = onnxruntime.InferenceSession(Config.vocoder_path)

    # 准备输入
    mel = mel.numpy()
    f0 = f0.astype(np.float32)
    mel = np.expand_dims(mel, axis=0).transpose(0, 2, 1)
    f0 = np.expand_dims(f0, axis=0)
    input_data = {
        'mel': mel,
        'f0': f0,
    }

    output = ort_session.run(['waveform'], input_data)[0]

    wave = output[0]
    return wave

def main(wave, mel_keyshift, speed, vocoder_keyshift):
    # 音频变速变调
    mel, f0 = wave_to_mel(wave, mel_keyshift, speed)
    wave = mel_to_wave(mel, f0, vocoder_keyshift, speed)
    return wave

if __name__ == '__main__':

    wave_path = r"path/to/audio.wav"
    wave, _ = sf.read(wave_path)


    # test1
    mel_keyshift = 0 
    speed = 1
    vocoder_keyshift = np.zeros(len(wave)//Config.hop_size) + 0

    wave_out = main(wave, mel_keyshift, speed, vocoder_keyshift)

    wave_path_opt = wave_path.replace('.wav', f'_melkeyshift{mel_keyshift}_speed{speed}_vocoderkeyshift{vocoder_keyshift[0]}.wav')
    sf.write(wave_path_opt, wave_out, Config.sample_rate)


    # test2
    mel_keyshift = 0 
    speed = 1.5
    vocoder_keyshift = np.zeros(len(wave)//Config.hop_size) + 0

    wave_out = main(wave, mel_keyshift, speed, vocoder_keyshift)

    wave_path_opt = wave_path.replace('.wav', f'_melkeyshift{mel_keyshift}_speed{speed}_vocoderkeyshift{vocoder_keyshift[0]}.wav')
    sf.write(wave_path_opt, wave_out, Config.sample_rate)


    # test3
    mel_keyshift = 6
    speed = 1
    vocoder_keyshift = np.zeros(len(wave)//Config.hop_size) + 0

    wave_out = main(wave, mel_keyshift, speed, vocoder_keyshift)

    wave_path_opt = wave_path.replace('.wav', f'_melkeyshift{mel_keyshift}_speed{speed}_vocoderkeyshift{vocoder_keyshift[0]}.wav')
    sf.write(wave_path_opt, wave_out, Config.sample_rate)

    # test4
    mel_keyshift = 0
    speed = 1
    vocoder_keyshift = np.zeros(len(wave)//Config.hop_size) + 6

    wave_out = main(wave, mel_keyshift, speed, vocoder_keyshift)

    wave_path_opt = wave_path.replace('.wav', f'_melkeyshift{mel_keyshift}_speed{speed}_vocoderkeyshift{vocoder_keyshift[0]}.wav')
    sf.write(wave_path_opt, wave_out, Config.sample_rate)

    # test5
    mel_keyshift = 0
    speed = 1
    vocoder_keyshift = np.zeros(len(wave)//Config.hop_size) + np.linspace(-12, 12, len(wave)//Config.hop_size)

    wave_out = main(wave, mel_keyshift, speed, vocoder_keyshift)

    wave_path_opt = wave_path.replace('.wav', f'_melkeyshift{mel_keyshift}_speed{speed}_vocoderkeyshift{vocoder_keyshift[0]}.wav')
    sf.write(wave_path_opt, wave_out, Config.sample_rate)


        











