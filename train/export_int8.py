# تصدير الموديل المدرَّب ONNX ثم ضغطه INT8 ونشره عبر الـWorker — بعد finetune.py.
# البيئة: WORKER + MUTQIN_KEY (بدونه تُطبع أوامر الرفع بدل التنفيذ).
import hashlib
import json
import os
import time

import nemo.collections.asr as nemo_asr

model = nemo_asr.models.ASRModel.restore_from('finetuned.nemo')
if hasattr(model, 'cur_decoder'):
    model.cur_decoder = 'ctc'  # فرع CTC هو الذي يستهلكه التطبيق (logprobs)
model.export('fc.onnx')  # يقبل log-mel جاهزًا: audio_signal [1,80,T] + length
print('ONNX:', os.path.getsize('fc.onnx') / 1e6, 'MB')

# INT8 ديناميكي لطبقات MatMul — نفس وصفة الموديل الحالي (٤× أصغر، دقة محفوظة)
from onnxruntime.quantization import QuantType, quantize_dynamic
quantize_dynamic('fc.onnx', 'fc-q8.onnx', weight_type=QuantType.QInt8, op_types_to_quantize=['MatMul'])
size = os.path.getsize('fc-q8.onnx')
sha1 = hashlib.sha1(open('fc-q8.onnx', 'rb').read()).hexdigest()
print('INT8:', size / 1e6, 'MB —', sha1)

# تقسيم مقاطع ≤٩٠م.ب (تحت حد جسم طلب الـWorker) + مانيفست بصيغة التطبيق
PART = 90 * 1024 * 1024
names = []
with open('fc-q8.onnx', 'rb') as f:
    i = 0
    while chunk := f.read(PART):
        name = f'fc-q8.part{i:02d}.bin'
        open(name, 'wb').write(chunk)
        names.append(name)
        i += 1

manifest = {
    'model': 'fastconformer-quran-q8', 'arch': 'FastConformer-Large hybrid (NVIDIA), 114M, INT8',
    'file': 'fc-q8.onnx', 'parts': len(names), 'partPrefix': 'fc-q8.part', 'partSuffix': '.bin',
    'size': size, 'sha1': sha1,
    'inputs': {'audio_signal': 'float32 [1,80,T] log-mel', 'length': 'int64 [1]'},
    'outputs': {'logprobs': '[1,T,1025] blank=1024'},
    'mel': {'sampleRate': 16000, 'features': 80, 'nFft': 512, 'winLength': 400, 'hopLength': 160,
            'preemph': 0.97, 'logGuard': 5.96e-8, 'normalize': 'per_feature'},
    'tokenizer': 'tokenizer.model (SentencePiece 1024 BPE)', 'version': 'v2',
}
json.dump(manifest, open('manifest.json', 'w'), ensure_ascii=False, indent=1)
print('مقاطع:', names)

# النشر عبر الـWorker: المقاطع أولًا والمانيفست آخر حاجة — فلا يرى أي تطبيق
# تحديثًا ناقص المقاطع. التطبيقات تكتشفه وتستأذن المستخدم قبل التنزيل.
WORKER = os.environ.get('WORKER', 'https://mutqin-collector.mutqin.workers.dev')
key = os.environ.get('MUTQIN_KEY')
order = [*names, 'manifest.json']
if key:
    import requests
    for name in order:
        for attempt in range(4):
            r = requests.post(f'{WORKER}/put', params={'key': f'model/{name}'},
                              headers={'x-mutqin-key': key}, data=open(name, 'rb').read())
            if r.ok and r.json().get('ok'):
                print('☁️ model/' + name)
                break
            time.sleep(15 * (attempt + 1))
        else:
            raise SystemExit(f'فشل رفع {name} — أعد تشغيل الخلية')
    print('تم النشر ✓')
else:
    print('\nمفيش MUTQIN_KEY — نفّذ بالترتيب:')
    for name in order:
        print(f"curl -X POST '{WORKER}/put?key=model%2F{name}' -H 'x-mutqin-key: $MUTQIN_KEY' --data-binary @{name}")
