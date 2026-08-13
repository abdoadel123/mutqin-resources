# فاين تيونينج موديل التسميع على أصوات المستخدمين — يتشغّل على Colab/Kaggle (GPU).
# الاستعمال: python finetune.py  (متغيرات اختيارية: DATA, EPOCHS, LR, BASE_REPO, HF_TOKEN)
# الخرج: finetuned.nemo — بعده شغّل export_int8.py
import glob, json, os, random

DATA = os.environ.get('DATA', '/content/train-out')
EPOCHS = int(os.environ.get('EPOCHS', '20'))
LR = float(os.environ.get('LR', '1e-5'))
BASE_REPO = os.environ.get('BASE_REPO', 'mohammed/fastconformer-quran-ar')

# ١) تقسيم المانيفست: ٩٠٪ تدريب / ١٠٪ تحقق، بمسارات مطلقة
rows = [json.loads(l) for l in open(f'{DATA}/manifest.jsonl', encoding='utf-8') if l.strip()]
assert rows, f'مفيش عيّنات في {DATA}/manifest.jsonl — شغّل prep.ts الأول'
for r in rows:
    r['audio_filepath'] = os.path.join(DATA, r['audio_filepath'])
random.seed(7)
random.shuffle(rows)
n_val = max(1, len(rows) // 10)

def dump(name, subset):
    with open(name, 'w', encoding='utf-8') as f:
        f.write('\n'.join(json.dumps(r, ensure_ascii=False) for r in subset))

dump('train_manifest.jsonl', rows[n_val:])
dump('val_manifest.jsonl', rows[:n_val])
print(f'عيّنات: {len(rows) - n_val} تدريب / {n_val} تحقق')

# ٢) تنزيل الشيكبوينت الأساسي (.nemo) من HuggingFace
from huggingface_hub import hf_hub_download, list_repo_files
nemo_file = next(f for f in list_repo_files(BASE_REPO) if f.endswith('.nemo'))
ckpt = hf_hub_download(BASE_REPO, nemo_file)
print('الأساس:', ckpt)

# ٣) التدريب — معدّل تعلّم منخفض كي لا ينسى الموديل القرآن العام
import lightning.pytorch as pl
import nemo.collections.asr as nemo_asr
from omegaconf import open_dict

model = nemo_asr.models.ASRModel.restore_from(ckpt)
common = {'sample_rate': 16000, 'batch_size': int(os.environ.get('BS', '8')),
          'num_workers': 2, 'pin_memory': True}
model.setup_training_data({'manifest_filepath': 'train_manifest.jsonl', 'shuffle': True,
                           'max_duration': 90.0, **common})
model.setup_validation_data({'manifest_filepath': 'val_manifest.jsonl', 'shuffle': False, **common})
with open_dict(model.cfg):
    model.cfg.optim.lr = LR
    if 'sched' in model.cfg.optim:
        model.cfg.optim.sched.min_lr = LR / 10
model.setup_optimization(model.cfg.optim)

trainer = pl.Trainer(devices=1, accelerator='gpu', max_epochs=EPOCHS,
                     precision='bf16-mixed', logger=False, enable_checkpointing=False,
                     log_every_n_steps=5)
model.set_trainer(trainer)
trainer.fit(model)

model.save_to('finetuned.nemo')
print('تم: finetuned.nemo — كمّل بـ export_int8.py')
