"""FastAPI API for the Vite CardChop workbench; reuses the original workers/data."""
import asyncio
from contextlib import asynccontextmanager
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import uuid

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from enhance_worker import validate_options, detector_status

STORAGE = Path(os.environ.get('STORAGE_DIR', ROOT / 'data')).resolve()
UPLOADS = STORAGE / 'uploads'
ENHANCED = STORAGE / 'enhanced'
REGISTRY = STORAGE / 'jobs.json'
DIST = ROOT / 'frontend' / 'dist'
jobs = {}
tasks = set()
enhancement_lock = asyncio.Lock()
ocr_slots = asyncio.Semaphore(2)
vlm_slots = asyncio.Semaphore(max(1, min(8, int(os.environ.get('VLM_CONCURRENCY', '3')))))
detector = {'available': False, 'reason': 'checking'}
provider = {'available': [], 'provider': None}
LIMITS = {'scale':(1,4), 'descratch':(0,1), 'denoise':(0,1), 'sharpen':(0,1), 'contrast':(0,.5)}
BOOLS = ('autoCrop', 'autoRotate', 'chromeOnly')


def public(card):
    return {k: card.get(k) for k in ('id','fileName','url','enhancedUrl','metadata','status','createdAt')} | {
        'workbench': card.get('workbench'), 'workbenchRevision': card.get('workbenchRevision', 0)}


def persist():
    records = [dict(c, status='queued' if c['status'] in ('processing','enhancing') else c['status']) for c in jobs.values()]
    tmp = REGISTRY.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(records, allow_nan=False))
    os.replace(tmp, REGISTRY)


def restore():
    for directory in (STORAGE, UPLOADS, ENHANCED): directory.mkdir(parents=True, exist_ok=True)
    if REGISTRY.exists():
        # Fail visibly on a corrupt registry; never overwrite it with an empty queue.
        records = json.loads(REGISTRY.read_text())
        if not isinstance(records, list): raise RuntimeError('Invalid jobs registry; original preserved')
        for record in records:
            original = Path(record.get('filePath', ''))
            if not original.is_file(): original = UPLOADS / Path(record.get('url','')).name
            if original.is_file():
                record['filePath'] = str(original)
                record.setdefault('metadata', {})
                if record.get('status') in ('processing','enhancing'): record['status']='queued'
                jobs[record['id']] = record
    for path in UPLOADS.iterdir():
        if not path.is_file() or path.stem in jobs: continue
        output = ENHANCED / (path.stem+'_enhanced.png')
        jobs[path.stem] = {'id':path.stem, 'fileName':path.name, 'filePath':str(path),
            'url':'/uploads/'+path.name, 'enhancedUrl':'/enhanced/'+output.name if output.exists() else None,
            'metadata':{}, 'status':'enhanced' if output.exists() else 'queued', 'createdAt':path.stat().st_mtime*1000}


async def worker(*args, timeout=180):
    proc = await asyncio.create_subprocess_exec(*map(str,args), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout)
    except (asyncio.CancelledError, asyncio.TimeoutError):
        if proc.returncode is None: proc.kill()
        await proc.wait()
        raise
    if proc.returncode: raise RuntimeError(stderr.decode(errors='replace')[-1000:] or stdout.decode(errors='replace')[-1000:] or 'Worker failed')
    result = json.loads(stdout)
    if not isinstance(result, dict) or result.get('error'): raise RuntimeError(result.get('error') if isinstance(result,dict) else 'Invalid worker result')
    return result


def schedule(coroutine):
    task = asyncio.create_task(coroutine)
    tasks.add(task)
    task.add_done_callback(tasks.discard)


async def read_card(card, vlm=False, opts=None):
    try:
        async with (vlm_slots if vlm else ocr_slots):
            if vlm:
                metadata = await worker('node', ROOT/'vlm_bridge.cjs', 'read', card['filePath'], json.dumps(opts or {}), timeout=180)
            else:
                metadata = await worker(sys.executable, ROOT/'ocr_worker.py', card['filePath'], timeout=90)
        card['metadata'].update(metadata)
        card['metadata'].pop('error',None)
        card['status'] = 'vlm_complete' if vlm else 'ocr_complete'
    except Exception as exc:
        card['metadata']['error'] = str(exc)[:1000]
        card['status'] = 'failed'
    finally: persist()


def validated_workbench(value):
    if not isinstance(value,dict) or set(value)-{'rotation','quad','customSettings'}: raise ValueError('Invalid workbench fields')
    custom=value.get('customSettings',{})
    if not isinstance(custom,dict) or set(custom)-set(LIMITS)-set(BOOLS): raise ValueError('Unsupported per-card setting')
    for key,v in custom.items():
        if key in BOOLS:
            if type(v) is not bool: raise ValueError(f'Invalid {key}')
        elif type(v) not in (int,float) or not math.isfinite(v) or not LIMITS[key][0]<=v<=LIMITS[key][1]: raise ValueError(f'Invalid {key}')
    geom=validate_options({'rotation':value.get('rotation',0), 'quad':value.get('quad')})
    return {'rotation':geom['rotation'], 'quad':geom['quad'], 'customSettings':custom}


def options(request, card):
    query=request.query_params
    if any(len(query.getlist(k))!=1 for k in query): raise ValueError('Repeated enhancement parameter')
    data={}
    for key in LIMITS:
        if key in query:
            if not query[key].strip(): raise ValueError(f'Invalid {key}')
            data[key]=float(query[key])
    for key in ('conservative','autoCrop','autoRotate'):
        if key in query:
            if query[key] not in ('true','false'): raise ValueError(f'Invalid {key}')
            data[key]=query[key]=='true'
    if 'rotation' in query:
        if query['rotation'] not in ('0','90','180','270'): raise ValueError('Invalid rotation')
        data['rotation']=int(query['rotation'])
    if 'quad' in query: data['quad']=json.loads(query['quad'])
    # Reject invalid global inputs even when a per-card override would replace them.
    data=validate_options(data)
    if card.get('workbench'):
        wb=validated_workbench(card['workbench']);custom=wb['customSettings']
        data.update(rotation=wb['rotation'],quad=wb['quad'])
        data.update({k:v for k,v in custom.items() if k in LIMITS or k in ('autoCrop','autoRotate')})
        if custom.get('chromeOnly'):
            m=card['metadata'];hay=' '.join(str(m.get(k,'')) for k in ('parallel','card_type','set_name','rarity','finish','foil','all_text','bio_text'))
            if not re.search(r'chrome|foil|refractor|prizm|holo|metallic|dazzler|parallel',hay,re.I): data['descratch']=0
    return validate_options(data)


def get_card(card_id):
    if card_id not in jobs: raise HTTPException(404, 'Card not found')
    return jobs[card_id]


def require_idle(card):
    if card['status'] in ('processing','enhancing'): raise HTTPException(409, 'Card is processing; wait for completion')


async def enhance_card(card, opts):
    try:
        async with enhancement_lock:
            output=ENHANCED/(card['id']+'_enhanced.png')
            result=await worker(sys.executable,ROOT/'enhance_worker.py',card['filePath'],output,json.dumps(opts))
            with output.open('rb') as file:
                if file.read(8)!=b'\x89PNG\r\n\x1a\n': raise RuntimeError('Worker output is not PNG')
            if not re.fullmatch(r'\d+x\d+', result.get('output_size','')) or type(result.get('yolo_cropped')) is not bool: raise RuntimeError('Incomplete enhancement result')
            card['metadata']['enhancement']=result
            for k in ('rotation','output_size','orientation','yolo_cropped','yolo_available','yolo_conf','detection_warning'):card['metadata'][k]=result.get(k)
            card['enhancedUrl']='/enhanced/'+output.name+'?v='+str(time.time_ns())
            card['status']='enhanced'
    except Exception as exc:
        card['metadata']['enhanceError']=str(exc)[:1000];card['status']='failed'
    finally: persist()


def begin_enhancement(card,opts):
    card['status']='enhancing';card['enhancedUrl']=None
    for key in ('enhancement','enhanceError','output_size','orientation'):card['metadata'].pop(key,None)
    persist();schedule(enhance_card(card,opts))


@asynccontextmanager
async def lifespan(app):
    global detector, provider
    restore()
    try: detector=await asyncio.to_thread(detector_status)
    except Exception as exc: detector={'available':False,'reason':str(exc)[:200]}
    try: provider=await worker('node',ROOT/'vlm_bridge.cjs','status',timeout=15)
    except Exception: provider={'available':[],'provider':None}
    yield
    for task in list(tasks): task.cancel()
    await asyncio.gather(*list(tasks),return_exceptions=True)
    persist()


app=FastAPI(title='CardChop',version='2.1.0',lifespan=lifespan)


@app.exception_handler(HTTPException)
async def api_error(request,exc):return JSONResponse({'error':exc.detail},status_code=exc.status_code)


@app.get('/health')
@app.get('/api/health')
async def health():
    return {'status':'ok','category':'wrestling','backend':'fastapi','frontend':'vite','revision':os.environ.get('RAILWAY_GIT_COMMIT_SHA','local'),
        'enhancement':{'available':True,'engine':'opencv','maxLongEdge':2200},'jobs':len(jobs),'yolo':detector,'vlm':provider,'review_gate':80}


@app.get('/api/cards')
async def cards():return {'cards':[public(c) for c in sorted(jobs.values(),key=lambda c:c.get('createdAt',0),reverse=True)]}


@app.get('/api/cards/export/json')
async def export():return JSONResponse([public(c) for c in jobs.values()],headers={'Content-Disposition':'attachment; filename=cardchop-audit.json'})


@app.post('/api/cards/upload')
async def upload(files:list[UploadFile]=File(...)):
    if not 1<=len(files)<=200:raise HTTPException(400,'Upload 1–200 images per request')
    created=[]; paths=[]
    try:
        for file in files:
            if not (file.content_type or '').startswith('image/'):raise HTTPException(400,'Only image files accepted')
            ext=Path(file.filename or 'scan.jpg').suffix.lower()
            if ext not in ('.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff'):raise HTTPException(400,'Unsupported image format')
            card_id=str(uuid.uuid4());path=UPLOADS/(card_id+ext);paths.append(path)
            count=0
            with path.open('wb') as target:
                while chunk:=await file.read(1024*1024):
                    count+=len(chunk)
                    if count>25*1024*1024:raise HTTPException(413,'Image exceeds 25 MB')
                    target.write(chunk)
            try:
                with Image.open(path) as image:image.verify()
            except Exception:raise HTTPException(400,'Cannot decode uploaded image')
            created.append({'id':card_id,'fileName':file.filename,'filePath':str(path),'url':'/uploads/'+path.name,'enhancedUrl':None,'metadata':{},'status':'processing','createdAt':time.time()*1000})
    except BaseException:
        for path in paths:path.unlink(missing_ok=True)
        raise
    finally:
        for file in files:await file.close()
    for card in created:jobs[card['id']]=card
    persist()
    for card in created:schedule(read_card(card))
    return {'cards':[public(c) for c in created]}


@app.post('/api/cards/vlm-all')
async def read_all(request:Request):
    candidates=[c for c in jobs.values() if c['status'] not in ('processing','enhancing')]
    only=request.query_params.get('only')
    if only=='failed':candidates=[c for c in candidates if c['status']=='failed']
    elif only=='pending':candidates=[c for c in candidates if c['status']!='vlm_complete' or c['metadata'].get('review_needed')]
    opts={k:request.query_params[k] for k in ('provider','model') if k in request.query_params}
    opts['escalate']=request.query_params.get('escalate')!='false'
    for card in candidates:card['status']='processing';schedule(read_card(card,True,opts))
    persist();return {'started':len(candidates)}


@app.post('/api/cards/enhance-all')
async def enhance_all(request:Request):
    candidates=[c for c in jobs.values() if c['status'] not in ('processing','enhancing')]
    try:work=[(c,options(request,c)) for c in candidates]
    except (ValueError,TypeError) as exc:raise HTTPException(400,str(exc))
    for card,opts in work:begin_enhancement(card,opts)
    return {'started':len(work)}


@app.get('/api/cards/{card_id}')
async def one(card_id:str):return {'card':public(get_card(card_id))}


@app.patch('/api/cards/{card_id}/workbench')
async def edit(card_id:str,request:Request):
    card=get_card(card_id);require_idle(card)
    try:body=await request.json()
    except ValueError:raise HTTPException(400,'Invalid JSON')
    if not isinstance(body,dict) or type(body.get('expectedRevision')) is not int or body['expectedRevision']!=card.get('workbenchRevision',0):raise HTTPException(409,'Card changed; reopen the editor')
    try:wb=validated_workbench(body.get('workbench'))
    except (ValueError,TypeError) as exc:raise HTTPException(400,str(exc))
    card.update(workbench=wb,workbenchRevision=card.get('workbenchRevision',0)+1,enhancedUrl=None,status='queued')
    for key in ('enhancement','enhanceError','output_size','orientation'):card['metadata'].pop(key,None)
    persist();return {'card':public(card)}


@app.post('/api/cards/{card_id}/enhance')
async def enhance(card_id:str,request:Request):
    card=get_card(card_id);require_idle(card)
    try:opts=options(request,card)
    except (ValueError,TypeError) as exc:raise HTTPException(400,str(exc))
    begin_enhancement(card,opts);return {'card':public(card)}


@app.post('/api/cards/{card_id}/vlm')
async def vlm(card_id:str,request:Request):
    card=get_card(card_id);require_idle(card)
    if not provider.get('available'):raise HTTPException(503,'No vision provider configured')
    card['status']='processing';persist()
    opts={k:request.query_params[k] for k in ('provider','model') if k in request.query_params}
    opts['escalate']=request.query_params.get('escalate')!='false'
    await read_card(card,True,opts)
    if card['status']=='failed':raise HTTPException(502,card['metadata'].get('error','Vision read failed'))
    return {'card':public(card)}


@app.delete('/api/cards/{card_id}')
async def delete(card_id:str):
    card=get_card(card_id);require_idle(card)
    Path(card['filePath']).unlink(missing_ok=True)
    (ENHANCED/(card_id+'_enhanced.png')).unlink(missing_ok=True)
    del jobs[card_id];persist();return {'ok':True}


# API paths always resolve before the Vite build; missing API routes return JSON 404.
@app.api_route('/api/{rest:path}',methods=['GET','POST','PATCH','DELETE'])
async def unknown_api(rest:str):raise HTTPException(404,'Unknown API route')


UPLOADS.mkdir(parents=True,exist_ok=True);ENHANCED.mkdir(parents=True,exist_ok=True)
app.mount('/uploads',StaticFiles(directory=UPLOADS),name='uploads')
app.mount('/enhanced',StaticFiles(directory=ENHANCED),name='enhanced')
if DIST.exists():app.mount('/',StaticFiles(directory=DIST,html=True),name='vite')
