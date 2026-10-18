# offline judge evaluation on a dump: for detections in the good-light frames and in the one-strip frames,
# the k nearest bank examples by cosine; prints how often the vote disagrees with the model, by segment and class
import glob, json, os, struct, sys, numpy as np
dump, bank_path = sys.argv[1], sys.argv[2]; K = 7
b = open(bank_path, 'rb').read(); nc, dim, n = struct.unpack('iii', b[4:16]); off = 16; cls = []; vec = []
for i in range(n):
    cls.append(struct.unpack('i', b[off:off+4])[0]); vec.append(np.frombuffer(b[off+4:off+4+4*dim], dtype=np.float32)); off += 4 + 4*dim
cls = np.array(cls); vec = np.stack(vec)
meta = json.load(open(os.path.join(dump, 'feats.json'))); shapes = [tuple(s) for s in meta['feats']]; W = meta['model_w']; strides = [W // s[3] for s in shapes]
seg = {'good': lambda f: f < 3600 or f >= 9000, 'onestrip': lambda f: 3600 <= f < 9000}
stats = {s: {'n': 0, 'disagree': 0, 'sims': [], 'shares': [], 'pairs': {}} for s in seg}
for fp in sorted(glob.glob(os.path.join(dump, '*.feats'))):
    fr = int(os.path.basename(fp)[:-6]); s = 'good' if seg['good'](fr) else 'onestrip'
    raw = np.fromfile(fp, dtype=np.float32); feats = []; o = 0
    for sh in shapes:
        k = int(np.prod(sh)); feats.append(raw[o:o+k].reshape(sh)); o += k
    for line in open(fp[:-6] + '.det'):
        v = line.split()
        if len(v) < 6 or float(v[1]) < 0.25: continue
        c, x1, y1, x2, y2 = int(v[0]), *(float(x) for x in v[2:6])
        side = max(x2-x1, y2-y1); k = 0 if side <= 96 else 1 if side <= 192 else 2; k = min(k, 2); st = strides[k]
        gx = min(max(int(((x1+x2)/2)//st), 0), shapes[k][3]-1); gy = min(max(int(((y1+y2)/2)//st), 0), shapes[k][2]-1)
        f = feats[k][0, :, gy, gx]; nrm = np.linalg.norm(f)
        if nrm == 0: continue
        q = f / nrm; sims = vec @ q; idx = np.argsort(-sims)[:K]; votes = np.bincount(cls[idx], minlength=nc); win = int(votes.argmax())
        st_ = stats[s]; st_['n'] += 1; st_['sims'].append(float(sims[idx[0]])); st_['shares'].append(float(votes[win]) / K)
        if win != c:
            st_['disagree'] += 1; key = f'{c}->{win}'; st_['pairs'][key] = st_['pairs'].get(key, 0) + 1
for s, st_ in stats.items():
    sims = np.array(st_['sims']); shares = np.array(st_['shares'])
    print(f"{s}: {st_['n']} detections, vote disagrees {st_['disagree']}, top-sim median {np.median(sims):.3f} (10th pct {np.percentile(sims,10):.3f}), unanimous {np.mean(shares==1):.0%}; pairs {st_['pairs']}")
