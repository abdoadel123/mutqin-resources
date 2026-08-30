# Phase-0 gates (STREAMING_PLAN.md §2.4): the same audio decoded three ways —
#   offline : model.transcribe over the full utterance — the reference reading
#   stream  : NeMo cache-aware streaming, chunk by chunk with carried caches — proves the
#             model actually streams (tokens arrive once and match the offline reading)
#   onnx    : the exported fc-stream.onnx over the full utterance — proves export fidelity
# stream≈offline AND onnx≈offline together prove the mechanics with zero training risk.
# Usage: python stream_parity.py model.nemo audio1.wav [audio2.wav ...]   (16 kHz mono)
import sys

import numpy as np
import soundfile
import torch

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

STREAM_GATE, ONNX_GATE = 0.05, 0.02  # WER against the offline reference

model_path, wavs = sys.argv[1], sys.argv[2:]
assert wavs, 'هات ملف wav واحد على الأقل (16kHz mono)'
model = nemo_asr.models.ASRModel.restore_from(model_path, map_location='cpu')
if hasattr(model, 'cur_decoder'):
    model.change_decoding_strategy(decoder_type='ctc')  # the branch the app ships
model.eval()


def wer(ref: str, hyp: str) -> float:
    a, b = ref.split(), hyp.split()
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[-1] / max(1, len(a))


def stream_one(path: str) -> str:
    buffer = CacheAwareStreamingAudioBuffer(model=model)
    buffer.append_audio_file(path, stream_id=-1)
    cache_channel, cache_time, cache_len = model.encoder.get_initial_cache_state(batch_size=1)
    previous_hypotheses, pred_out_stream, text = None, None, ''
    for chunk_audio, chunk_lengths in buffer:
        with torch.inference_mode():
            (pred_out_stream, transcribed, cache_channel, cache_time, cache_len,
             previous_hypotheses) = model.conformer_stream_step(
                processed_signal=chunk_audio, processed_signal_length=chunk_lengths,
                cache_last_channel=cache_channel, cache_last_time=cache_time,
                cache_last_channel_len=cache_len,
                keep_all_outputs=buffer.is_buffer_empty(),
                previous_hypotheses=previous_hypotheses, previous_pred_out=pred_out_stream,
                return_transcription=True)
        text = getattr(transcribed[0], 'text', transcribed[0])
    return text


def onnx_one(path: str) -> str:
    import onnxruntime
    session = onnx_one.session
    if session is None:
        session = onnx_one.session = onnxruntime.InferenceSession('fc-stream.onnx')
    audio, rate = soundfile.read(path, dtype='float32')
    assert rate == 16000, f'{path}: {rate}Hz — المطلوب 16kHz'
    with torch.inference_mode():
        mel, mel_len = model.preprocessor(
            input_signal=torch.tensor(audio)[None], length=torch.tensor([len(audio)]))
    cache_channel, cache_time, cache_len = model.encoder.get_initial_cache_state(batch_size=1)
    feeds = {'audio_signal': mel.numpy(), 'length': mel_len.numpy().astype(np.int64),
             'cache_last_channel': cache_channel.numpy(), 'cache_last_time': cache_time.numpy(),
             'cache_last_channel_len': cache_len.numpy().astype(np.int64)}
    wanted = {i.name for i in session.get_inputs()}
    missing, extra = wanted - feeds.keys(), feeds.keys() - wanted
    assert not missing and not extra, f'مدخلات الجراف تغيّرت: ناقص {missing} زائد {extra}'
    logprobs = session.run(['logprobs'], feeds)[0][0]  # [frames, vocab+1]
    blank, ids, prev = logprobs.shape[1] - 1, [], -1
    for token in logprobs.argmax(axis=1):
        if token != prev and token != blank:
            ids.append(int(token))
        prev = token
    return model.tokenizer.ids_to_text(ids)


onnx_one.session = None
worst_stream = worst_onnx = 0.0
for path in wavs:
    offline = model.transcribe([path], batch_size=1)[0]
    offline = getattr(offline, 'text', offline)
    streamed, exported = stream_one(path), onnx_one(path)
    stream_wer, onnx_wer = wer(offline, streamed), wer(offline, exported)
    worst_stream, worst_onnx = max(worst_stream, stream_wer), max(worst_onnx, onnx_wer)
    print(f'\n== {path}\noffline: {offline}\nstream : {streamed}   (wer {stream_wer:.1%})'
          f'\nonnx   : {exported}   (wer {onnx_wer:.1%})')

print(f'\nالبوابة: stream≤{STREAM_GATE:.0%} طلعت {worst_stream:.1%} · onnx≤{ONNX_GATE:.0%} طلعت {worst_onnx:.1%}')
if worst_stream <= STREAM_GATE and worst_onnx <= ONNX_GATE:
    print('✅ الميكانيكا عدّت — المتبقي على التدريب سؤال الدقة وحده (المرحلة ٢)')
else:
    sys.exit('❌ فشل التكافؤ — عيب تصدير/بث، لا عيب تدريب: يُفهم قبل صرف أي دولار على GPU')
