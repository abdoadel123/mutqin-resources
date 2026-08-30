# Exports a cache-aware streaming model to ONNX INT8 — the streaming counterpart of
# ../export_int8.py. No candidate heads: v3/v3b died with the aligner (STREAMING_PLAN.md §5-b),
# so the graph ships logprobs only and the Gather fragility never exists.
# Also dumps stream_meta.json: the chunk geometry and cache shapes the device adapter
# (plugin stateful session, plan §3.1) will need — read from the model, never assumed.
# Usage: python export_stream.py [model.nemo]   (default: finetuned.nemo)
import json
import os
import sys

import nemo.collections.asr as nemo_asr

src = sys.argv[1] if len(sys.argv) > 1 else 'finetuned.nemo'
model = nemo_asr.models.ASRModel.restore_from(src, map_location='cpu')
if hasattr(model, 'cur_decoder'):
    model.cur_decoder = 'ctc'  # the app consumes the CTC branch
model.eval()

encoder_cfg = model.cfg.encoder
assert encoder_cfg.get('att_context_style') == 'chunked_limited', \
    'ليس موديلًا حدثيًا — درِّب بـ STREAMING=1 أولًا (finetune.py)'

# Cache tensors become explicit inputs/outputs, so the app can carry them across chunks
# in native memory instead of re-hearing the window.
model.set_export_config({'cache_support': 'True'})
model.export('fc-stream.onnx')
print('ONNX:', round(os.path.getsize('fc-stream.onnx') / 1e6, 1), 'MB')

# Geometry for the phase-1 adapter, read from the model itself.
cache_channel, cache_time, cache_len = model.encoder.get_initial_cache_state(batch_size=1)
att_context = list(encoder_cfg.att_context_size)
subsampling = int(encoder_cfg.subsampling_factor)
meta = {
    'att_context_size': att_context,
    # One encoder frame = subsampling x window_stride seconds (0.08s at 8 x 10ms).
    'encoder_frame_seconds': subsampling * float(model.cfg.preprocessor.window_stride),
    'chunk_encoder_frames': att_context[1] + 1,
    'chunk_mel_frames': (att_context[1] + 1) * subsampling,
    'cache_last_channel_shape': list(cache_channel.shape),
    'cache_last_time_shape': list(cache_time.shape),
    'cache_last_channel_len_shape': list(cache_len.shape),
    'sample_rate': int(model.cfg.preprocessor.sample_rate),
    'n_mels': int(model.cfg.preprocessor.features),
    'vocab': model.decoder.num_classes_with_blank if hasattr(model.decoder, 'num_classes_with_blank') else None,
}
with open('stream_meta.json', 'w') as f:
    json.dump(meta, f, indent=1)
print('stream_meta.json:', meta)

from onnxruntime.quantization import QuantType, quantize_dynamic

quantize_dynamic('fc-stream.onnx', 'fc-stream-q8.onnx',
                 weight_type=QuantType.QInt8, op_types_to_quantize=['MatMul'])
print('INT8:', round(os.path.getsize('fc-stream-q8.onnx') / 1e6, 1), 'MB')
print('تم: fc-stream-q8.onnx + stream_meta.json — شغّل stream_parity.py قبل أي نشر')
