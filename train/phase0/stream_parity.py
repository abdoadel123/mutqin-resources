# Phase-0 gates (STREAMING_PLAN.md §2.4): the same audio decoded three ways —
#   offline : model.transcribe over the full utterance — the reference reading
#   stream  : NeMo cache-aware streaming, chunk by chunk with carried caches — proves the
#             model actually streams (tokens arrive once and match the offline reading)
#   onnx    : the exported fc-stream.onnx driven CHUNK BY CHUNK with carried caches — the
#             graph is the chunk step (feeding it a whole utterance emits one chunk's text:
#             measured 91.3% WER on the pod, 2026-08-30), exactly as the device adapter calls it
# stream≈offline AND onnx≈offline together prove the mechanics with zero training risk.
# Usage: python stream_parity.py model.nemo audio1.wav [audio2.wav ...]   (16 kHz mono)
import json
import sys

import numpy as np
import soundfile
import torch

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

STREAM_GATE, ONNX_GATE = 0.05, 0.05  # WER against the offline reference (chunked onnx pays the
# same latency-context price the stream leg pays, so it shares its gate)

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
    """NeMo's own cache-aware stream, one clip. A clip that throws is reported, not fatal:
    the run's whole point is the comparison across the gate set, and losing the remaining
    gates to one bad clip is how a $0.5 measurement answered two of six questions (31/08)."""
    try:
        return _stream_one(path)
    except Exception as e:  # noqa: BLE001 — one clip must not cost the other five
        return f'<stream-error: {type(e).__name__}: {e}>'


def _stream_one(path: str) -> str:
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
    meta = json.load(open('stream_meta.json'))
    chunk = int(meta['chunk_mel_frames'])
    audio, rate = soundfile.read(path, dtype='float32')
    assert rate == 16000, f'{path}: {rate}Hz — المطلوب 16kHz'
    with torch.inference_mode():
        mel, _ = model.preprocessor(
            input_signal=torch.tensor(audio)[None], length=torch.tensor([len(audio)]))
    mel = mel.numpy()
    outs = [o.name for o in session.get_outputs()]
    lp_name = next(o for o in outs if 'logprob' in o)
    channel_next = next(o for o in outs if 'channel' in o and 'len' not in o)
    time_next = next(o for o in outs if 'time' in o)
    len_next = next(o for o in outs if 'len' in o)
    # get_initial_cache_state is layers-first; the graph is batch-first (measured 2026-08-30).
    cache_channel, cache_time, cache_len = model.encoder.get_initial_cache_state(batch_size=1)
    cache_channel = cache_channel.numpy().swapaxes(0, 1)
    cache_time = cache_time.numpy().swapaxes(0, 1)
    cache_len = cache_len.numpy().astype(np.int64)
    ids, prev, blank = [], -1, None
    for off in range(0, mel.shape[2], chunk):
        piece = mel[:, :, off:off + chunk]
        n = piece.shape[2]
        if n < chunk:
            piece = np.pad(piece, ((0, 0), (0, 0), (0, chunk - n)))
        feeds = {'audio_signal': piece.astype(np.float32), 'length': np.array([n], dtype=np.int64),
                 'cache_last_channel': cache_channel, 'cache_last_time': cache_time,
                 'cache_last_channel_len': cache_len}
        res = dict(zip(outs, session.run(outs, feeds)))
        cache_channel, cache_time = res[channel_next], res[time_next]
        cache_len = res[len_next].astype(np.int64)
        logprobs = res[lp_name][0]
        if blank is None:
            blank = logprobs.shape[1] - 1
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
