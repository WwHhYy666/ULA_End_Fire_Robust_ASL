import os
import numpy as np
from scipy.io.wavfile import write

# 配置参数
PCM_DIR = "C:\\Users\\25401\\Desktop\\acoustic\\ssl\\data_exp\\pcm_audio"
WAV_DIR = os.path.join(os.path.dirname(PCM_DIR), "wav_audio")
SAMPLE_RATE = 16000
NUM_CHANNELS = 6
DTYPE = np.int16  # 假设是16位PCM

def pcm_to_wav(pcm_file_path):
    with open(pcm_file_path, 'rb') as f:
        raw_data = f.read()

    # 计算样本数（每个样本2字节 * 6通道）
    bytes_per_sample = np.dtype(DTYPE).itemsize
    frame_size_bytes = bytes_per_sample * NUM_CHANNELS
    total_samples = len(raw_data) // frame_size_bytes
    remainder = len(raw_data) % frame_size_bytes

    if remainder:
        print(f"⚠️ 警告: {pcm_file_path} 数据长度不完整，已截断 {remainder} 字节后继续转换。")
        raw_data = raw_data[: total_samples * frame_size_bytes]

    # 将原始数据转为 int16 数组
    audio_array = np.frombuffer(raw_data, dtype=DTYPE)

    # 重塑为 (samples, channels)
    audio_array = audio_array.reshape((total_samples, NUM_CHANNELS))

    # 输出 WAV 文件路径
    wav_file_name = os.path.splitext(os.path.basename(pcm_file_path))[0] + ".wav"
    wav_file_path = os.path.join(WAV_DIR, wav_file_name)

    # 写入 WAV 文件
    write(wav_file_path, SAMPLE_RATE, audio_array)
    print(f"✅ 已转换: {pcm_file_path} -> {wav_file_path}")

def main():
    os.makedirs(WAV_DIR, exist_ok=True)

    for filename in os.listdir(PCM_DIR):
        if filename.lower().endswith('.pcm'):
            full_path = os.path.join(PCM_DIR, filename)
            pcm_to_wav(full_path)

if __name__ == "__main__":
    main()