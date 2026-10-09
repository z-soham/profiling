import gzip,sys,re,collections
f=sys.argv[1]
samples=[]  # (tid,time,frames[(sym,dso)])
cur=None
for line in gzip.open(f,'rt'):
    if not line.strip():
        if cur: samples.append(cur); cur=None
        continue
    if not line.startswith(('\t',' \t')) and '/' in line.split(':')[0]:
        m=re.match(r'\s*(\d+)/(\d+)\s+([\d.]+):',line)
        if m: cur=[int(m.group(2)),float(m.group(3)),[],int(m.group(1))]
    elif cur is not None:
        p=line.split(None,1)
        if len(p)<2: continue
        rest=p[1].strip()
        m=re.match(r'(.*) \((.*)\)$',rest)
        if m: cur[2].append((m.group(1),m.group(2)))
if cur: samples.append(cur)
pid=samples[0][3]; print('samples',len(samples),'main tid',pid)
def cls(fr):
    if not fr: return 'none'
    s,d=fr[0]
    if 'libomp' in d or 'libiomp' in d or 'libgomp' in d: return 'spin'
    if 'perf-' in d and '.map' in d: return 'zen_jit'
    if 'libzendnnl' in d or 'aocl' in d.lower(): return 'zen_lib'
    if 'libggml-cpu' in d: return 'ggml_cpu:'+s.split('(')[0][:40]
    if 'libggml-base' in d: return 'ggml_base'
    if 'libggml-zendnn' in d: return 'ggml_zendnn:'+s[:40]
    if 'libc.' in d or 'tcmalloc' in d: return 'libc/malloc:'+s[:30]
    if 'kernel' in d: return 'kernel'
    return 'other:'+d.split('/')[-1][:20]
# main-thread leaf histogram
mh=collections.Counter(cls(s[2]) for s in samples if s[0]==pid)
tot=sum(mh.values()); print('MAIN THREAD leaf, % of its samples'); 
for k,v in mh.most_common(14): print(f'  {100*v/tot:5.1f}  {k}')
# per 2ms bin: main-thread class, #threads spin / #threads work
bins=collections.defaultdict(lambda: collections.defaultdict(list))
for t,tm,fr,_ in samples: bins[int(tm*500)][t].append(cls(fr))
agg=collections.defaultdict(lambda:[0,0,0,0])
for b,th in bins.items():
    if pid not in th: continue
    mc=th[pid][0]; key='spin' if mc=='spin' else ('zen' if mc.startswith('zen') else ('ggml_zendnn' if mc.startswith('ggml_zendnn') else mc.split(':')[0]))
    sp=sum(1 for t,c in th.items() if t!=pid and c[0]=='spin'); wk=sum(1 for t,c in th.items() if t!=pid and c[0]!='spin')
    a=agg[key]; a[0]+=1; a[1]+=sp; a[2]+=wk; a[3]+=len(th)-1
print('2ms bins by main-thread class: n_bins, avg other-threads spinning, avg other-threads working, avg sampled')
for k,a in sorted(agg.items(),key=lambda x:-x[1][0])[:10]: print(f'  {k:14s} {a[0]:6d}  spin {a[1]/a[0]:5.1f}  work {a[2]/a[0]:5.1f}  of {a[3]/a[0]:5.1f}')
