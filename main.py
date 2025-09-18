import onnxruntime
import os
from pathlib import Path
from util.f0Analyzer import F0Analyzer,resample_align_curve
from util.wav2mel import PitchAdjustableMelSpectrogram
import dataclasses
import soundfile as sf
import numpy as np
import torch
import yaml

@dataclasses.dataclass
class Config:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        self.sample_rate = config['sample_rate']
        self.win_size = config['win_size']
        self.hop_size = config['hop_size']
        self.n_mels = config['n_mels']
        self.n_fft = config['n_fft']
        self.mel_fmin = config['mel_fmin']
        self.mel_fmax = config['mel_fmax']
        self.f0_extractor = config['f0_extractor']
        self.f0_min = config['f0_min']
        self.f0_max = config['f0_max']

        self.vocoder_path = config['vocoder_path']
        self.wave_path = config['wave_path']
        self.input_dir = config.get('input_dir', 'input')
        self.output_dir = config.get('output_dir', 'output')

        self.mel_keyshift = config['mel_keyshift']
        self.speed = config['speed']
        self.vocoder_keyshift = config['vocoder_keyshift']
        self.batch_mode = config.get('batch_mode', False)

def dynamic_range_compression_torch(x, C=1, clip_val=1e-9):
    return torch.log(torch.clamp(x, min=clip_val) * C)
def wave_to_mel(wave,mel_keyshift,speed):
    '''
    wave shape=(n_samples,)
    mel_keyshift: float
    speed: float,                        default:1.0
    '''
    wave = torch.from_numpy(wave).float()
    melAnalysis = PitchAdjustableMelSpectrogram(
        sample_rate=config.sample_rate, 
        n_fft=config.n_fft, 
        win_length=config.win_size, 
        hop_length=config.hop_size, 
        f_min=config.mel_fmin, 
        f_max=config.mel_fmax,
        n_mels=config.n_mels
        )
    mel = melAnalysis(
        wave.unsqueeze(0),
        mel_keyshift,
        speed,
        ).squeeze()
    
    mel = dynamic_range_compression_torch(mel)
    
    f0Analyzer = F0Analyzer(
        sampling_rate = config.sample_rate,
        f0_extractor = config.f0_extractor,
        hop_size = config.hop_size,
        f0_min = config.f0_min,
        f0_max = config.f0_max
        )
    
    f0, _ = f0Analyzer(wave, n_frames=mel.shape[1],speed=speed)
    
    # Base on mel_keyshift adjust f0
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
    timestep = config.hop_size/config.sample_rate
    vocoder_keyshift = resample_align_curve(
        vocoder_keyshift, 
        timestep, 
        timestep*speed, 
        mel.shape[1]
    ) # Adjust vocoder keyshift to mel length
    f0 = f0*(2 ** (vocoder_keyshift / 12))

    # load vocodee
    ort_session = onnxruntime.InferenceSession(config.vocoder_path)

    # Prepare input
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
    # Audio speed_shift and vocoder_keyshift
    mel, f0 = wave_to_mel(wave, mel_keyshift, speed)
    wave = mel_to_wave(mel, f0, vocoder_keyshift, speed)
    return wave

if __name__ == '__main__':
    # Load config
    config = Config('config.yaml')
    
    if config.batch_mode:
        # Create output directory if it doesn't exist
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Get all .wav files in the input directory
        input_files = list(Path(config.input_dir).glob('*.wav'))
        
        for wav_path in input_files:
            try:
                # Read wave
                wave, _ = sf.read(str(wav_path))
                
                # render
                mel_keyshift = config.mel_keyshift
                speed = config.speed
                vocoder_keyshift = config.vocoder_keyshift 
                vocoder_keyshift = np.zeros(len(wave)//config.hop_size) + vocoder_keyshift
                
                wave_out = main(wave, mel_keyshift, speed, vocoder_keyshift)
                
                # Generate output filename
                output_filename = f"{wav_path.stem}_melkeyshift_{mel_keyshift}_speed_{speed}_vocoderkeyshift_{vocoder_keyshift[0]}.wav"
                output_path = os.path.join(config.output_dir, output_filename)
                
                # Save output wave
                sf.write(output_path, wave_out, config.sample_rate)
                print(f"Finish: {wav_path.name}")
                
            except Exception as e:
                print(f"Error when processing {wav_path.name}: {str(e)}")
                continue
    else:        
        # Batch mode false
        wave, _ = sf.read(config.wave_path)

        # render
        mel_keyshift = config.mel_keyshift
        speed = config.speed
        vocoder_keyshift = config.vocoder_keyshift 
        vocoder_keyshift = np.zeros(len(wave)//config.hop_size) + vocoder_keyshift

        wave_out = main(wave, mel_keyshift, speed, vocoder_keyshift)

        wave_path_opt = config.wave_path.replace('.wav', f'_melkeyshift_{mel_keyshift}_speed_{speed}_vocoderkeyshift_{vocoder_keyshift[0]}.wav')
        sf.write(wave_path_opt, wave_out, config.sample_rate)

        











