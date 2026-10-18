#!/usr/bin/env python3
# HDMI page for the Q, http://localhost:8082/ : one pane per running edgeble-infer instance
# (its stream port and its results folder), each with a green/red banner from its last result.
# /?panes=a  shows one pane full screen; /?panes=a,b,c,d the 2x2 grid (default: whichever run).
import http.server, json, os, glob, urllib.parse
PORT = 8082
H = '/home/arduino/jagan'
PANES = {'a': (8080, 'EXISTING LINE', 'model as installed'),
         'b': (8090, 'EXISTING LINE + JUDGE', 'corrected on the spot'),
         'c': (8100, 'AFTER RETRAIN', 'head retrained and swapped'),
         'd': (8110, 'JUDGE + RETRAIN', 'the cause is gone'),
         'l': (0, 'LOAD', 'Hexagon NPU / CPU / RAM / temperature, one line a second from tools/load.sh')}
import re as _re
def load_line():
    # tools/load.sh > ~/jagan/load.txt writes one \r-terminated record a second; return the last one, escapes stripped
    try:
        with open(os.path.join(H, 'load.txt'), 'rb') as f:
            f.seek(0, 2); n = f.tell(); f.seek(max(0, n - 4000)); raw = f.read().decode('utf-8', 'ignore')
        recs = [r for r in _re.split(r'[\r\n]', raw) if r.strip()]
        return _re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', recs[-1]) if recs else 'load.sh not running'
    except Exception as e:
        return 'load.sh not running'
PAGE = '''<!doctype html><html><head><meta charset="utf-8"><title>Edgeble QC</title><style>
html,body{margin:0;height:100%%;background:#111;color:#fff;font-family:sans-serif;overflow:hidden}
#g{display:grid;grid-template-columns:%s;grid-template-rows:%s;width:100vw;height:100vh;gap:4px;background:#000}
.p{position:relative;background:#000;overflow:hidden}.p img{width:100%%;height:100%%;object-fit:contain;display:block}
.t{position:absolute;left:0;right:0;top:0;padding:.5vh 1vw;font-size:2.6vh;font-weight:700;background:rgba(0,0,0,.65)}
.b{position:absolute;left:0;right:0;bottom:0;padding:.6vh 1vw;font-size:3.4vh;font-weight:700;display:flex;justify-content:space-between;align-items:center}
.b small{font-size:2.2vh;font-weight:400}.go{background:rgba(26,156,60,.92)}.ng{background:rgba(198,40,40,.92)}.idle{background:rgba(60,60,60,.9)}
</style></head><body><div id="g">%s</div><script>
const P=%s;
async function tick(){for(const p of P){try{const r=await fetch('/status/'+p+'?'+Date.now());const s=await r.json();
const b=document.getElementById('b'+p);b.className='b '+(s.result==='GO'?'go':s.result==='NG'?'ng':'idle');
b.innerHTML='<span>'+(s.result==='GO'?'OK ✓':s.result==='NG'?'NG ✗':'WAITING')+'</span><small>'+(s.detail||'')+' &nbsp; GO '+(s.go||0)+' / NG '+(s.ng||0)+'</small>';}catch(e){}}}
setInterval(tick,500);tick();
/* MJPEG stream per tile (the engine's multipart stream); on a dropped connection, reconnect after 1 s */
for(const im of document.querySelectorAll('img[data-port]')){const u='http://localhost:'+im.dataset.port+'/';const go=()=>{im.src=u+'?'+Date.now();};im.onerror=()=>setTimeout(go,1000);go();}
</script></body></html>'''
def pane(n):
    port, title, sub = PANES[n]
    if n == 'l':
        return f'<div class="p" style="display:flex;align-items:center;justify-content:center"><pre id="load" style="font-size:2.6vh;line-height:1.8;white-space:pre-wrap;padding:2vw;margin:0;color:#8f8">...</pre><div class="t">{title} <small>{sub}</small></div></div>'
    return f'<div class="p"><img id="i{n}" data-port="{port}" src=""><div class="t">{title} <small>{sub}</small> <span id="j{n}" style="float:right;color:#8f8"></span></div><div class="b idle" id="b{n}"><span>WAITING</span><small></small></div></div>'
def running():
    return [n for n in PANES if os.path.isdir(os.path.join(H, f'reports-{n}'))]
def page(names):
    cols = '1fr' if len(names) == 1 else '1fr 1fr'; rows = '1fr' if len(names) <= 2 else '1fr 1fr'
    return PAGE % (cols, rows, ''.join(pane(n) for n in names), json.dumps(names))
def flips(name):
    p = os.path.join(H, f'reports-{name}', 'judge.log')
    try: return sum(1 for _ in open(p))
    except OSError: return None
def last_result(name):
    files = sorted(glob.glob(os.path.join(H, f'reports-{name}', '*', 'logs', 'results_*.log')))
    if not files: return {'result': None, 'flips': flips(name)}
    try:
        with open(files[-1], 'rb') as f:
            f.seek(0, 2); n = f.tell(); f.seek(max(0, n - 4096)); tail = f.read().decode('utf-8', 'ignore').strip().splitlines()
    except OSError: return {'result': None}
    for line in reversed(tail):
        if 'RESULT:' in line:
            parts = [p.strip() for p in line.split('|')]
            d = {'time': parts[0][11:19], 'result': 'GO' if 'RESULT:GO' in line else 'NG'}
            for p in parts:
                if p.startswith('GO:'): d['go'] = int(p[3:])
                if p.startswith('NG:'): d['ng'] = int(p[3:])
            ll = [p for p in parts if p.startswith('LL:')]; lr = [p for p in parts if p.startswith('LR:')]
            d['detail'] = (ll[0] if ll else '') + '  ' + (lr[0] if lr else '')
            d['flips'] = flips(name)
            return d
    return {'result': None, 'flips': flips(name)}
class Hd(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        u = urllib.parse.urlparse(self.path); q = urllib.parse.parse_qs(u.query)
        if u.path == '/load':
            b = load_line().encode(); ct = 'text/plain; charset=utf-8'
        elif u.path.startswith('/status/'):
            b = json.dumps(last_result(u.path[8:])).encode(); ct = 'application/json'
        else:
            names = [n for n in q.get('panes', [''])[0].split(',') if n in PANES] or running() or ['a']
            b = page(names).encode(); ct = 'text/html; charset=utf-8'
        self.send_response(200); self.send_header('Content-Type', ct); self.send_header('Content-Length', str(len(b))); self.end_headers(); self.wfile.write(b)
http.server.ThreadingHTTPServer(('0.0.0.0', PORT), Hd).serve_forever()
