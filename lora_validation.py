"""Fail-closed split, image, adapter and embedding checks for the LoRA pipeline."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from lora_models import file_hash

CLASSES = ['normal', 'bacteria', 'virus', 'COVID-19']
BRANCHES = ['densenet', 'raddino', 'chexfound', 'chexzero']


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_frame(data, split):
    name = {'train':'train_data.csv', 'val':'val_data.csv', 'test':'test_data_sample.csv'}[split]
    path = Path(data)/name
    df = pd.read_csv(path, dtype={'new_filename':'string'})
    if df.columns.tolist() != ['new_filename'] + CLASSES or df.empty:
        raise ValueError(f'Wrong/empty CSV: {path}')
    names = df.new_filename
    if names.isna().any() or names.duplicated().any():
        raise ValueError(f'Missing/duplicate names: {path}')
    for name in names:
        if Path(name).name != name or '/' in name or '\\' in name:
            raise ValueError(f'Unsafe image name: {name}')
    if split != 'test':
        y = df[CLASSES].to_numpy(dtype=float)
        if not np.isfinite(y).all() or not np.isin(y,[0,1]).all() or not (y.sum(1)==1).all():
            raise ValueError(f'Invalid labels: {path}')
        if not (y.sum(0)>0).all():
            raise ValueError(f'All four classes must be present: {path}')
    return df


def audit(data, patient_map=None, strict_split_checks=False):
    """Recompute instead of trusting an old audit; no filename-derived patient IDs."""
    data = Path(data)
    frames = {s: read_frame(data,s) for s in ('train','val','test')}
    seen_names, seen_pixels = {}, {}
    fingerprints, counts, collisions = {}, {}, []
    for split, df in frames.items():
        rows = []
        for name in df.new_filename.astype(str):
            path = data / f'{split}_images' / name
            if not path.is_file():
                raise FileNotFoundError(path)
            with Image.open(path) as img:
                img = img.convert('RGB')
                h=hashlib.sha256(str(img.size).encode()); h.update(img.tobytes()); pixel_hash=h.hexdigest()
            if name in seen_names and seen_names[name] != split:
                collisions.append(f'filename {name}: {seen_names[name]} / {split}')
            if pixel_hash in seen_pixels and seen_pixels[pixel_hash][0] != split:
                collisions.append(f'identical decoded image: {seen_pixels[pixel_hash]} / {(split,name)}')
            seen_names[name] = split
            seen_pixels[pixel_hash] = (split,name)
            rows.append([name, file_hash(path), pixel_hash])
        labels = df[CLASSES].to_numpy(dtype=int).tolist() if split != 'test' else None
        fingerprints[split] = digest({'rows': rows, 'labels': labels, 'classes':CLASSES})
        counts[split] = dict(zip(CLASSES, df[CLASSES].sum().astype(int).tolist())) if split != 'test' else {'images':len(df)}
    if collisions and strict_split_checks:
        raise ValueError('Cross-split leakage detected (curate splits before training):\n'+'\n'.join(collisions[:20]))
    patient_status = 'UNVERIFIED: no patient metadata; filename/pixel checks cannot establish patient independence'
    if patient_map:
        p = pd.read_csv(patient_map, dtype=str)
        if set(p.columns) != {'split','new_filename','patient_id'} or p.isna().any().any():
            raise ValueError('patient-map needs exactly split,new_filename,patient_id with no missing values')
        if p.duplicated(['split','new_filename']).any() or (p.patient_id.str.strip()=='').any():
            raise ValueError('Duplicate mapping rows or empty patient IDs')
        expected = {(s,str(n)) for s,d in frames.items() for n in d.new_filename}
        if set(zip(p.split,p.new_filename)) != expected:
            raise ValueError('patient-map must cover precisely every image in all three splits')
        if (p.groupby('patient_id')['split'].nunique()>1).any():
            raise ValueError('Patient overlap across splits')
        patient_status = 'verified from supplied metadata, SHA256='+file_hash(patient_map)
    report = {'schema':2, 'classes':CLASSES, 'splits':fingerprints, 'counts':counts,
              'cross_split_matches':collisions,
              'patient_status':patient_status}
    report['id'] = digest(report)
    return report


def checked_cache(cache, branch, split, data, require_manifest=False, audit_report=None):
    path = Path(cache)/branch/f'{split}.npz'
    with np.load(path, allow_pickle=False) as d:
        x, y, names = d['x'], d['y'], d['names'].astype(str)
        if str(d['branch']) != branch:
            raise ValueError(f'Wrong branch: {path}')
        if 'size' not in d or int(d['size']) < 1:
            raise ValueError(f'Missing/invalid image size: {path}')
        metadata = json.loads(str(d['metadata'])) if 'metadata' in d else None
    df = read_frame(data,split)
    expected_y = df[CLASSES].to_numpy().argmax(1).astype(np.int64) if split != 'test' else np.empty(0,dtype=np.int64)
    if not np.array_equal(names,df.new_filename.astype(str)) or not np.array_equal(y,expected_y):
        raise ValueError(f'Cache rows/labels differ from CSV (one view required): {path}')
    if y.dtype!=np.int64 or y.ndim!=1 or x.ndim!=2 or x.shape[0]!=len(df) or x.shape[1]<1 or x.dtype!=np.float32 or not np.isfinite(x).all():
        raise ValueError(f'Invalid feature matrix: {path}')
    if require_manifest:
        if metadata is None or metadata.get('classes') != CLASSES or metadata.get('branch') != branch:
            raise ValueError(f'Cache has no valid provenance: {path}')
        if metadata.get('split_fingerprint') != audit_report['splits'][split]:
            raise ValueError(f'Stale cache images or labels: {path}')
        if metadata.get('audit_id') != audit_report['id']:
            raise ValueError(f'Cache belongs to a different training split/patient audit: {path}')
        if metadata.get('feature_sha256') != hashlib.sha256(x.tobytes()).hexdigest():
            raise ValueError(f'Corrupted features: {path}')
    return x,y,names,metadata


def save_cache(path, x, y, names, branch, size, metadata):
    # pandas 3 string columns may export object arrays; keep NPZ pickle-free.
    names=np.asarray(names,dtype=str)
    y=np.asarray(y,dtype=np.int64)
    x=np.asarray(x,dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite embeddings')
    metadata = dict(metadata, classes=CLASSES, branch=branch,
                    feature_sha256=hashlib.sha256(x.tobytes()).hexdigest())
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite cache: {path}')
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix('.tmp.npz')
    np.savez_compressed(temp, x=x, y=y, names=names, branch=branch, size=size,
                        metadata=json.dumps(metadata,sort_keys=True))
    temp.replace(path)


def curate(data, destination):
    """Create a new copy, keeping test fixed and preferring test > val > train."""
    import shutil
    data, destination = Path(data).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f'Curated destination must be new: {destination}')
    if destination == data or data in destination.parents:
        raise ValueError('Curated destination must be outside source data directory')
    frames = {s:read_frame(data,s) for s in ('test','val','train')}
    seen = {}; retained = {}; excluded=[]
    # Test labels are never used for curation. Removal uses image identity only.
    for split, df in frames.items():
        keep=[]
        for i,name in enumerate(df.new_filename.astype(str)):
            with Image.open(data/f'{split}_images'/name) as image:
                image=image.convert('RGB')
                h=hashlib.sha256(str(image.size).encode()); h.update(image.tobytes()); key=h.hexdigest()
            if key in seen and split!='test':
                excluded.append({'split':split,'new_filename':name,'duplicate_of_split':seen[key][0],
                                 'duplicate_of':seen[key][1],'pixel_sha256':key})
            else:
                keep.append(i)
                seen[key]=(split,name)
        retained[split]=df.iloc[keep].copy()
        if split!='test' and not (retained[split][CLASSES].sum()>0).all():
            raise ValueError('Curation removed a whole class; a new grouped split is needed')
    # Do not modify/hard-link source images: a curated image can be edited independently.
    destination.mkdir(parents=True)
    for split, df in retained.items():
        filename={'train':'train_data.csv','val':'val_data.csv','test':'test_data_sample.csv'}[split]
        df.to_csv(destination/filename,index=False)
        folder=destination/f'{split}_images'; folder.mkdir()
        for name in df.new_filename.astype(str):
            shutil.copy2(data/f'{split}_images'/name,folder/name)
    pd.DataFrame(excluded,columns=['split','new_filename','duplicate_of_split','duplicate_of','pixel_sha256']).to_csv(
        destination/'excluded_duplicates.csv',index=False)
    result=audit(destination,strict_split_checks=True)
    result['curation']={'policy':'decoded RGB exact duplicates; test > val > train; deduplicate within train/val',
                        'source':str(data),'excluded':len(excluded),'retained':{s:len(d) for s,d in retained.items()},
                        'limitation':'Does not detect near duplicates or infer patient identity'}
    (destination/'curation_report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result
