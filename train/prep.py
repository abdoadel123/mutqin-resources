# تجهيز داتا التدريب — يتشغّل في Colab: يسحب جلسات المساهمة كلها من الـWorker
# (أحداث + صوت + كلمات الصفحات)، يغلّف الصوت WAV، ويبني manifest.jsonl لNeMo.
# النص: كلمات المصحف حيث أصاب القارئ، وما قاله فعلًا حيث أخطأ (علامات الجلسة).
# البيئة: WORKER (رابط الـworker) + MUTQIN_KEY — واختياريًا MIN_ACC وOUT.
import json
import os

import requests

WORKER = os.environ.get('WORKER', 'https://mutqin-collector.mutqin.workers.dev')
H = {'x-mutqin-key': os.environ['MUTQIN_KEY']}
MIN_ACC = float(os.environ.get('MIN_ACC', '0.85'))
SR = 16000
OUT = os.environ.get('OUT', '/content/train-out')
os.makedirs(OUT, exist_ok=True)


def list_all(prefix=''):
    r = requests.get(f'{WORKER}/data/list', params={'prefix': prefix}, headers=H)
    r.raise_for_status()
    return r.json()


def get(key):
    r = requests.get(f'{WORKER}/data/get', params={'key': key}, headers=H)
    r.raise_for_status()
    return r.content


# كلمات الصفحات (مولَّدة من محرّك التطبيق) — بيانات ثابتة من هذا الريبو نفسه
PAGES_URL = 'https://raw.githubusercontent.com/abdoadel123/mutqin-resources/main/train/pages.json'
PAGES = requests.get(PAGES_URL).json()


def expected(page):  # الجلسة قد تمتد صفحات — نمدّ المتوقَّع ثلاث صفحات
    out = []
    for p in range(page, min(605, page + 3)):
        out += PAGES.get(str(p), [])
    return out


def build_text(page, marks, said):
    exp = expected(page)
    words, settled, correct = [], 0, 0
    for i, m in enumerate(marks):
        if m == 0 or i >= len(exp):
            continue  # كلمة لم تُبلغ (توقف مبكر أو تلميح)
        settled += 1
        if m == 1:
            correct += 1
            words.append(exp[i])
        else:
            words.append(said[i] if i < len(said) and said[i] else exp[i])
    return ' '.join(words), settled, correct


def wav_header(n):
    import struct
    return struct.pack('<4sI4s4sIHHIIHH4sI', b'RIFF', 36 + n, b'WAVE', b'fmt ', 16, 1, 1, SR, SR * 2, 2, 16, b'data', n)


objs = list_all()
audio = {o['key'].rsplit('/', 1)[-1]: o['key'] for o in objs if o['key'].startswith('audio/')}
manifest, skipped = [], 0
for o in (x for x in objs if x['key'].startswith('events/')):
    batch = json.loads(get(o['key']))
    for ev in batch.get('events', []):
        d = ev.get('d') or {}
        if ev.get('k') != 'contribAudio' or not d.get('path'):
            continue
        name = d['path'].rsplit('/', 1)[-1]
        if name not in audio:
            print('⚠️ صوت مفقود:', name)
            continue
        meta = d.get('meta') or {}
        text, settled, correct = build_text(d.get('page', 1), meta.get('marks', []), meta.get('said', []))
        acc = correct / settled if settled else 0
        if acc < MIN_ACC or not text:
            skipped += 1
            continue
        pcm = get(audio[name])
        wname = name.replace('.pcm', '.wav')
        with open(os.path.join(OUT, wname), 'wb') as f:
            f.write(wav_header(len(pcm)) + pcm)
        manifest.append({'audio_filepath': wname, 'duration': round(len(pcm) / 2 / SR, 2), 'text': text,
                         'page': d.get('page'), 'device': batch.get('device'), 'accuracy': round(acc, 3)})
        print(f'✓ {wname} — {settled} كلمة، دقّة {acc:.0%}')

with open(os.path.join(OUT, 'manifest.jsonl'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(json.dumps(r, ensure_ascii=False) for r in manifest))
print(f'\nتم: {len(manifest)} عيّنة في {OUT} (استُبعد {skipped} بدقّة أقل من {MIN_ACC})')
