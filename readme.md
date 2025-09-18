# Mel-Wav

基于openvpi的pc-nsf-hifigan声码器对音频（.wav）进行变调、变速、以及重新合成。

## 功能

  * **变调（Pitch Shifting）**: 调整音频的音高，同时保持语速不变。
  * **变速（Time Stretching）**: 调整音频的播放速度，同时保持音高不变。
  * **音高调整**: 支持对梅尔频谱和声码器分别进行音高调整，详见example.png。
  * **批处理**: 支持对整个文件夹的 .wav 文件进行批量处理。

## 环境要求

使用 pip 安装依赖：

```bash
pip install torch numpy onnxruntime pyyaml soundfile parselmouth pyworld
```

## 使用方法

### 1\. 准备声码器模型

[声码器下载]([https://github.com/openvpi/SingingVocoders](https://github.com/openvpi/vocoders/releases/tag/pc-nsf-hifigan-44.1k-hop512-128bin-2025.02))。请将模型文件放置在项目目录中，并在 `config.yaml` 文件中指定其路径。

### 2\. 配置 `config.yaml`

创建 `config.yaml` 文件（可从 `config.default.yaml` 复制并重命名），并修改其中的参数。

#### 参数说明：

  * **`f0_extractor`**: F0提取算法，可选 "parselmouth" 或 "harvest"。
  * **`vocoder_path`**: 声码器模型的路径。
  * **`input_dir`**: （批处理模式）输入文件夹的路径。
  * **`output_dir`**: （批处理模式）输出文件夹的路径。
  * **`wave_path`**: （单文件模式）输入 .wav 文件的路径。
  * **`mel_keyshift`**: 梅尔频谱的音高偏移（半音）。
  * **`speed`**: 播放速度。
  * **`vocoder_keyshift`**: 声码器的音高偏移（半音）。
  * **`batch_mode`**: 是否启用批处理模式（`true` 或 `false`）。

### 3\. 运行

#### 单文件处理

1.  在 `config.yaml` 中设置 `batch_mode: false`。
2.  在 `wave_path` 中指定要处理的 .wav 文件的路径。
3.  运行 `main.py`：
    ```bash
    python main.py
    ```

处理后的文件将会在原始文件名的基础上，添加处理参数作为后缀，并保存在与原始文件相同的目录中。

#### 批处理

1.  在 `config.yaml` 中设置 `batch_mode: true`。
2.  在 `input_dir` 中指定包含 .wav 文件的输入文件夹。
3.  在 `output_dir` 中指定处理后文件的输出文件夹。
4.  运行 `main.py`：
    ```bash
    python main.py
    ```

程序将会处理 `input_dir` 中的所有 .wav 文件，并将结果保存在 `output_dir` 中。

## 致谢

  *  `wav2mel.py` 来自 [openvpi/SingingVocoders](https://github.com/openvpi/SingingVocoders)。
  *  `pc-nsf-hifigan` 来自 [openvpi/vocoders](https://github.com/openvpi/vocoders)。
  *  原repo作者 [yjzxkxdn](https://github.com/yjzxkxdn)
