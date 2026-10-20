
const API={
  cards:'/api/cards',
  one:id=>`/api/cards/${id}`,
  upload:'/api/cards/upload',
  vlm:id=>`/api/cards/${id}/vlm`,
  enhance:id=>`/api/cards/${id}/enhance`,
  image:c=>c.enhancedUrl||c.url||`/uploads/${encodeURIComponent(c.fileName||'')}`,
};
const DEMO=new URLSearchParams(location.search).get('demo')==='1';
const DEMO_CARDS=[
  ['card-0193.jpg','/demo/card-01.jpg',1501,2090,false],
  ['card-0196.jpg','/demo/card-02.jpg',1504,2094,false],
  ['card-0197.jpg','/demo/card-03.jpg',1502,2100,false],
  ['card-0198.jpg','/demo/card-04.jpg',1501,2092,false],
  ['card-0199.jpg','/demo/card-05.jpg',2728,1928,true],
  ['card-0200.jpg','/demo/card-06.jpg',2711,1923,true],
  ['card-0206.jpg','/demo/card-07.jpg',1505,2097,false],
  ['card-0209.jpg','/demo/card-08.jpg',1506,2102,false],
  ['card-0212.jpg','/demo/card-09.jpg',1502,2101,false],
  ['card-0214.jpg','/demo/card-10.jpg',1492,2092,false],
  ['card-0217.jpg','/demo/card-11.jpg',1501,2090,false],
  ['card-0232.jpg','/demo/card-12.jpg',2720,1930,true],
].map((x,i)=>({
  id:`demo-${i+1}`,fileName:x[0],url:x[1],enhancedUrl:null,status:'queued',
  metadata:{demoWidth:x[2],demoHeight:x[3],orientation:x[4]?'landscape':'portrait'},
  createdAt:Date.now()-i*1000
}));

const State=CardChopState;
const batch=new State.BatchRun();
const storage={get(key){try{return JSON.parse(localStorage.getItem(key))}catch{return null}},set(key,value){try{localStorage.setItem(key,JSON.stringify(value))}catch{}}};
const session=storage.get('cardchop.workbench.session')||{};
let cards=[];
let selected=new Set(session.remaining?.length?session.remaining:(session.selected||[]));
let running=new Set();
let natural=new Map();
let dragDepth=0;
let toastTimer=null;
let loadVersion=0;
let currentModal=null;
let previousFocus=null;
let editing=null;
let editImage=null;
let logs=Array.isArray(session.logs)?session.logs.slice(-250):[];
const $=id=>document.getElementById(id);
const settings=State.settingsFrom(storage.get('cardchop.exact.settings'));
function persistSession(){storage.set('cardchop.workbench.session',{selected:[...selected],remaining:batch.busy||batch.remaining.length?batch.remaining:(session.remaining||[]),logs})}
function saveSettings(){storage.set('cardchop.exact.settings',settings)}
function clamp01(n){return Math.max(0,Math.min(1,n))}
function q(extra={}){
  const opts=extra.options||State.effectiveOptions(extra.card,settings);
  return new URLSearchParams(Object.entries(opts).filter(([,v])=>v!==null).map(([k,v])=>[k,Array.isArray(v)?JSON.stringify(v):String(v)])).toString();
}
function log(entry){logs.push({timestamp:new Date().toISOString(),...entry});logs=logs.slice(-250);persistSession()}
function esc(s){return String(s??'').replace(/[&<>"]/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[ch]))}
function fmt(n,d=2){return Number(n).toFixed(d)}
function statusOf(c){
  const s=String(c.status||'queued').toLowerCase();
  if(/fail|error/.test(s))return'failed';
  if(/process|enhancing|running/.test(s))return'processing';
  if(c.enhancedUrl||/enhanced/.test(s))return'enhanced';
  if(/vlm|ocr|complete|done/.test(s))return'read';
  return'queued';
}
function progress(){
  if(!cards.length)return 0;
  const done=cards.filter(c=>['enhanced','failed'].includes(statusOf(c))).length;
  return Math.round(done/cards.length*100);
}
function cardDims(c){
  if(c.metadata?.demoWidth)return [c.metadata.demoWidth,c.metadata.demoHeight];
  if(c.metadata?.output_size && /^\d+x\d+$/.test(c.metadata.output_size)){
    return c.metadata.output_size.split('x').map(Number);
  }
  if(c.metadata?.enhancement?.output_size && /^\d+x\d+$/.test(c.metadata.enhancement.output_size)){
    return c.metadata.enhancement.output_size.split('x').map(Number);
  }
  return natural.get(c.id)||[null,null];
}
function landscape(c){
  if(c.metadata?.orientation==='landscape')return true;
  const [w,h]=cardDims(c);
  return w&&h ? w/h>1.12 : false;
}
function cardBadge(c){
  const s=statusOf(c);
  if(running.has(c.id)||s==='processing')return'…';
  if(s==='enhanced')return'✓';
  if(s==='failed')return'!';
  return'';
}
function render(){
  const focusedId=document.activeElement?.closest('.card')?.dataset.id;
  selected=new Set([...selected].filter(id=>cards.some(c=>c.id===id)));
  const enhanced=cards.filter(c=>statusOf(c)==='enhanced').length;
  $('summary').textContent=`${cards.length} scans · ${selected.size} selected · ${enhanced} enhanced`;
  $('runSelected').textContent=`Run on ${selected.size} selected`;
  $('runSelected').disabled=DEMO||!selected.size||batch.busy;
  $('stopBatch').classList.toggle('hidden',!batch.busy);
  $('stopBatch').disabled=batch.stopping;
  $('batchStatus').textContent=batch.busy?`${batch.stopping?'Stopping after current card · ':''}${batch.ids.length-batch.remaining.length}/${batch.ids.length} handled`: `${cards.filter(c=>statusOf(c)==='failed').length} failed · ${DEMO?'fixture only':session.remaining?.length?'Recovered queue — Run resumes after reload':'Ready'}`;
  for(const key of Object.keys(State.DEFAULTS))if($(key))$(key).disabled=batch.busy;
  $('resumeBatch').disabled=batch.busy;
  persistSession();
  $('selectAll').disabled=!cards.length;
  $('clearSelection').disabled=!selected.size;
  const pct=progress();
  $('progressBar').style.width=pct+'%';
  $('progressText').textContent=pct+'%';

  if(!cards.length){
    $('grid').innerHTML=`<button class="empty" id="emptyAdd"><strong>Drop wrestling-card scans here</strong><span>JPG · PNG · WEBP<br>or click to choose files</span></button>`;
    $('emptyAdd').onclick=()=>$('filePicker').click();
    return;
  }
  $('grid').innerHTML=cards.map(c=>{
    const sel=selected.has(c.id);
    const [w,h]=cardDims(c);
    const dims=w&&h?`${w} × ${h}${landscape(c)?' landscape':''}`:'dimensions loading…';
    return `<article class="card ${sel?'selected':''} ${landscape(c)?'landscape':''}" data-id="${esc(c.id)}" tabindex="0" aria-selected="${sel}">
      <div class="imagebox">
        <span class="check"></span>
        ${cardBadge(c)?`<span class="badge">${esc(cardBadge(c))}</span>`:''}
        <img src="${esc(API.image(c))}" alt="${esc(c.fileName)}" loading="lazy" data-card-image="${esc(c.id)}">
      </div>
      <div class="meta">
        <div class="filename">${esc(c.fileName)}</div>
        <div class="dims">${esc(dims)}</div>
        <div class="card-tools"><button data-action="edit" aria-label="Edit ${esc(c.fileName)}">Edit${c.workbench?' *':''}</button><button data-action="inspect" aria-label="Inspect ${esc(c.fileName)}">Inspect</button></div>
      </div>
    </article>`;
  }).join('');
  document.querySelectorAll('.card').forEach(el=>{
    const id=el.dataset.id;
    el.onclick=()=>toggle(id);
    el.ondblclick=e=>{e.preventDefault();e.stopPropagation();openCompare(id)};
    el.querySelectorAll('[data-action]').forEach(button=>{button.onclick=e=>{e.stopPropagation();button.dataset.action==='edit'?void openEditor(id):openInspect(id)};button.onkeydown=e=>e.stopPropagation();button.ondblclick=e=>e.stopPropagation()});
    el.onkeydown=e=>{
      if(e.key==='Enter'||e.key===' '){e.preventDefault();toggle(id)}
      if(e.key==='o'||e.key==='O'){e.preventDefault();openCompare(id)}
      if(e.key==='e'||e.key==='E'){e.preventDefault();void openEditor(id)}
      if(e.key==='i'||e.key==='I'){e.preventDefault();openInspect(id)}
    };
  });
  if(focusedId)Array.from(document.querySelectorAll('.card')).find(el=>el.dataset.id===focusedId)?.focus();
  document.querySelectorAll('[data-card-image]').forEach(img=>{
    img.onload=()=>{
      const id=img.dataset.cardImage;
      const c=cards.find(x=>x.id===id);
      if(!c||c.metadata?.demoWidth||c.metadata?.output_size)return;
      const size=[img.naturalWidth,img.naturalHeight];
      const prev=natural.get(id);
      if(!prev||prev[0]!==size[0]||prev[1]!==size[1]){
        natural.set(id,size);
        const meta=img.closest('.card')?.querySelector('.dims');
        if(meta)meta.textContent=`${size[0]} × ${size[1]}${size[0]/size[1]>1.12?' landscape':''}`;
        if(size[0]/size[1]>1.12)img.closest('.card')?.classList.add('landscape');
      }
    };
  });
}
function toggle(id){
  if(selected.has(id))selected.delete(id);else selected.add(id);
  updateSelection();
}
function updateSelection(){
  document.querySelectorAll('.card').forEach(el=>{const active=selected.has(el.dataset.id);el.classList.toggle('selected',active);el.setAttribute('aria-selected',String(active))});
  $('summary').textContent=`${cards.length} scans · ${selected.size} selected · ${cards.filter(c=>statusOf(c)==='enhanced').length} enhanced`;
  $('runSelected').textContent=`Run on ${selected.size} selected`;
  $('runSelected').disabled=DEMO||!selected.size||batch.busy;
  $('clearSelection').disabled=!selected.size;persistSession();
}
function showToast(msg,ms=2600){
  clearTimeout(toastTimer);
  $('toast').textContent=msg;
  $('toast').classList.add('on');
  toastTimer=setTimeout(()=>$('toast').classList.remove('on'),ms);
}
async function json(url,init){
  const r=await fetch(url,init);
  const text=await r.text();
  let body={};
  try{body=text?JSON.parse(text):{}}catch{throw new Error(`Invalid JSON response (${r.status})`)}
  if(!r.ok){const err=new Error(body.error||text.slice(0,180)||`${r.status} ${r.statusText}`);err.status=r.status;throw err}
  return body;
}
async function loadCards(){
  if(DEMO){cards=DEMO_CARDS;if(!selected.size)selected=new Set(cards.slice(0,8).map(c=>c.id));render();return}
  const version=++loadVersion;
  try{
    const body=await json(API.cards);
    if(version!==loadVersion)return;
    const next=body.cards||body;
    if(!Array.isArray(next))throw new Error('Invalid cards response');
    cards=next;render();
  }catch(err){showToast(`Backend unavailable: ${err.message}`,4500)}
}
async function waitUntilIdle(id,timeout=45000){
  const start=Date.now();
  while(Date.now()-start<timeout){
    const body=await json(API.one(id));
    const c=body.card||body;
    cards=cards.map(x=>x.id===id?c:x);
    render();
    if(!['processing','enhancing'].includes(String(c.status||'').toLowerCase()))return c;
    await new Promise(r=>setTimeout(r,850));
  }
  throw new Error('Timed out waiting for OCR/VLM');
}
function needsVlm(c){
  const m=c.metadata||{};
  return !(m.player_name||m.wrestler_name||m.card_name||m.set_name||m.year);
}
async function runCard(id,options){
  let c=cards.find(x=>x.id===id);
  if(!c)return;
  running.add(id);render();
  try{
    if(String(c.status).toLowerCase()==='processing')c=await waitUntilIdle(id);
    if(needsVlm(c)&&!['processing','enhancing'].includes(String(c.status).toLowerCase())){
      try{
        const body=await json(API.vlm(id),{method:'POST'});
        c=body.card||c;
        cards=cards.map(x=>x.id===id?c:x);
        render();
      }catch(err){
        if(err.status!==409)console.warn('VLM read skipped:',err);
      }
    }
    if(['processing','enhancing'].includes(String(c.status||'').toLowerCase()))c=await waitUntilIdle(id);
    await json(`${API.enhance(id)}?${q({card:c,options})}`,{method:'POST'});
    c=await waitUntilIdle(id,190000);
    if(c.status!=='enhanced'||!c.enhancedUrl)throw new Error(c.metadata?.enhanceError||c.metadata?.error||'No completed output was returned');
  }finally{
    running.delete(id);render();
  }
}
async function runSelected(){
  if(DEMO){showToast('Fixture mode is read-only; no enhancement was performed.');return}
  const ids=[...selected];
  if(!batch.start(ids,settings))return;
  session.remaining=[];
  const resume=$('resumeBatch').checked;
  const options=new Map(cards.filter(c=>ids.includes(c.id)).map(c=>[c.id,State.effectiveOptions(c,batch.settings)]));
  persistSession();render();
  let ok=0,failed=0,skipped=0;
  try{
    for(const id of ids){
      if(batch.stopping)break;
      const c=cards.find(x=>x.id===id);
      if(!c){batch.complete(id);continue}
      if(resume&&State.resumable(c,options.get(id))){skipped++;batch.complete(id);log({id,fileName:c.fileName,status:'skipped',options:options.get(id)});render();continue}
      try{
        await runCard(id,options.get(id));ok++;
        const done=cards.find(x=>x.id===id);
        log({id,fileName:c.fileName,status:'enhanced',enhancedUrl:done?.enhancedUrl,result:done?.metadata?.enhancement});
      }catch(err){failed++;log({id,fileName:c.fileName,status:'failed',error:err.message});showToast(`${c.fileName}: ${err.message}`,4200)}
      batch.complete(id);persistSession();render();
    }
  }finally{
    const stopped=batch.stopping;
    batch.finish();session.remaining=[...batch.remaining];persistSession();await loadCards();render();
    showToast(`${ok} enhanced · ${skipped} resumed · ${failed} failed${stopped?` · ${batch.remaining.length} pending`:''}`,4500);
  }
}
async function uploadFiles(files){
  if(DEMO){showToast('Demo mode is read-only. Remove ?demo=1 to upload.');return}
  if(batch.busy){showToast('Stop or finish this batch before adding scans.');return}
  const list=[...files].filter(f=>f.type.startsWith('image/')||/\.(jpe?g|png|webp)$/i.test(f.name));
  if(!list.length)return;
  const added=[];
  try{
    for(let i=0;i<list.length;i+=200){
      const fd=new FormData();list.slice(i,i+200).forEach(f=>fd.append('files',f));
      showToast(`Uploading ${i+1}–${Math.min(i+200,list.length)} of ${list.length}…`,8000);
      const body=await json(API.upload,{method:'POST',body:fd});
      if(!Array.isArray(body.cards))throw new Error('Invalid upload response');
      added.push(...body.cards.map(c=>c.id));
    }
    await loadCards();selected=new Set(added);render();showToast(`${added.length} scans added`);
  }catch(err){await loadCards();if(added.length)selected=new Set(added);render();showToast(`${added.length} added · ${err.message}`,5000)}
}
function download(name,blobOrText,type='application/octet-stream'){
  const blob=blobOrText instanceof Blob?blobOrText:new Blob([blobOrText],{type});
  const a=document.createElement('a');
  a.href=URL.createObjectURL(blob);a.download=name;document.body.appendChild(a);a.click();a.remove();
  setTimeout(()=>URL.revokeObjectURL(a.href),3000);
}
async function downloadEnhanced(){
  if(DEMO){showToast('Fixture mode has no processed outputs.');return}
  const chosen=(selected.size?cards.filter(c=>selected.has(c.id)):cards).filter(c=>c.enhancedUrl&&c.status==='enhanced');
  if(!chosen.length){showToast('No enhanced PNGs yet — run the batch first.');return}
  // Intentionally no CDN/JSZip dependency: each completed output is the real server-rendered PNG.
  let n=0;
  for(const c of chosen){
    try{
      const r=await fetch(c.enhancedUrl);
      if(!r.ok)throw new Error(`${r.status}`);
      const blob=await r.blob();
      const base=(c.fileName||c.id).replace(/\.[^.]+$/,'');
      download(`${base}_${c.id.slice(0,8)}_enhanced.png`,blob);
      n++;
      await new Promise(r=>setTimeout(r,120));
    }catch(err){console.warn('download failed',c.fileName,err)}
  }
  showToast(`${n} enhanced PNG${n===1?'':'s'} downloaded`);
}
function auditPayload(){
  return cards.map(c=>{
    const [width,height]=cardDims(c);
    return {
      id:c.id,
      fileName:c.fileName,
      status:statusOf(c),
      selected:selected.has(c.id),
      sourceUrl:c.url,
      enhancedUrl:c.enhancedUrl,
      dimensions:width&&height?{width,height}:null,
      requestedSettings:{...settings},
      appliedSettings:c.metadata?.enhancement?.options||null,
      workbench:c.workbench||null,
      workbenchRevision:c.workbenchRevision||0,
      fixture:DEMO,
      metadata:c.metadata||{},
      createdAt:c.createdAt||null,
    };
  });
}
function exportAudit(){
  download('cardchop-audit.json',JSON.stringify(auditPayload(),null,2),'application/json');
}
function openCompare(id){
  const c=cards.find(x=>x.id===id);if(!c)return;
  $('compareTitle').textContent=c.fileName||'Compare';
  const [w,h]=cardDims(c);
  $('compareMeta').textContent=w&&h?`${w} × ${h}`:statusOf(c);
  $('compareOriginal').src=c.url||API.image(c);
  $('compareEnhanced').src=c.enhancedUrl||c.url||API.image(c);
  $('compareNotice').textContent=c.enhancedUrl?'Original / processed pixels':'No processed output yet; both panels show the original.';
  showModal('compareModal');
}
function closeCompare(){hideModal('compareModal')}

function bindSlider(id,key,formatter){
  const el=$(id),out=$(id+'Value');
  el.value=settings[key];
  const sync=()=>{
    settings[key]=Number(el.value);
    const min=Number(el.min),max=Number(el.max);
    el.style.setProperty('--pct',`${clamp01((settings[key]-min)/(max-min))*100}%`);
    out.textContent=formatter(settings[key]);
    saveSettings();
  };
  el.oninput=sync;sync();
}
bindSlider('scale','scale',v=>`${Number(v).toFixed(Number.isInteger(v)?0:1)}×`);

bindSlider('descratch','descratch',v=>fmt(v));
bindSlider('denoise','denoise',v=>fmt(v));
bindSlider('sharpen','sharpen',v=>fmt(v));
bindSlider('contrast','contrast',v=>fmt(v));
$('chromeOnly').checked=Boolean(settings.chromeOnly);
$('chromeOnly').onchange=()=>{settings.chromeOnly=$('chromeOnly').checked;saveSettings()};

$('selectAll').onclick=()=>{cards.forEach(c=>selected.add(c.id));updateSelection()};
$('clearSelection').onclick=()=>{selected.clear();updateSelection()};
$('runSelected').onclick=()=>void runSelected();
$('stopBatch').onclick=()=>{batch.stop();render()};
$('downloadEnhanced').onclick=()=>void downloadEnhanced();
$('exportAudit').onclick=exportAudit;
$('exportLog').onclick=()=>download('enhancement_log.jsonl',logs.map(e=>JSON.stringify(e)).join('\n')+'\n','application/x-ndjson');
$('addScans').onclick=()=>$('filePicker').click();
$('filePicker').onchange=e=>{if(e.target.files?.length)void uploadFiles(e.target.files);e.target.value=''};
$('compareClose').onclick=closeCompare;
$('compareModal').onclick=e=>{if(e.target===$('compareModal'))closeCompare()};

document.addEventListener('keydown',e=>{
  if(e.key==='Escape'&&currentModal){hideModal(currentModal);return}
  if(currentModal){trapModal(e);return}
  if(batch.busy&&/^[ar]$/i.test(e.key))return;
  if(e.target&&/INPUT|TEXTAREA|SELECT/.test(e.target.tagName))return;
  if(e.key==='a'||e.key==='A'){e.preventDefault();$('filePicker').click()}
  if(e.key==='r'||e.key==='R'){e.preventDefault();if(selected.size)void runSelected()}
  if(e.key==='c'||e.key==='C'){e.preventDefault();selected.clear();updateSelection()}
});

for(const ev of ['dragenter','dragover']){
  window.addEventListener(ev,e=>{
    e.preventDefault();
    if(ev==='dragenter')dragDepth++;
    $('dropOverlay').classList.add('on');
  });
}
window.addEventListener('dragleave',e=>{
  e.preventDefault();
  dragDepth=Math.max(0,dragDepth-1);
  if(!dragDepth)$('dropOverlay').classList.remove('on');
});
window.addEventListener('drop',e=>{
  e.preventDefault();dragDepth=0;$('dropOverlay').classList.remove('on');
  if(e.dataTransfer?.files?.length)void uploadFiles(e.dataTransfer.files);
});

void loadCards();
if(!DEMO){
  setInterval(()=>{
    if(document.hidden)return;
    if(!batch.busy&&cards.some(c=>['processing','enhancing'].includes(String(c.status||'').toLowerCase())))void loadCards();
  },2500);
}

function showModal(id){
  if(currentModal)$(currentModal).classList.remove('on');
  else previousFocus=document.activeElement;
  currentModal=id;$('app').inert=true;$(id).classList.add('on');
  $(id).querySelector('button,input,select')?.focus();
}
function hideModal(id){
  $(id).classList.remove('on');
  if(currentModal===id){currentModal=null;$('app').inert=false;previousFocus?.focus();}
  if(id==='editModal'){editing=null;editImage=null;}
}
function trapModal(event){
  if(event.key!=='Tab')return;
  const nodes=[...$(currentModal).querySelectorAll('button:not(:disabled),input:not(:disabled),select:not(:disabled),[tabindex="0"]')].filter(el=>el.getClientRects().length);
  if(!nodes.length)return;
  if(event.shiftKey&&document.activeElement===nodes[0]){event.preventDefault();nodes.at(-1).focus()}
  else if(!event.shiftKey&&document.activeElement===nodes.at(-1)){event.preventDefault();nodes[0].focus()}
}
function openInspect(id){
  const c=cards.find(c=>c.id===id);if(!c)return;
  $('inspectTitle').textContent=c.fileName;
  $('inspectData').textContent=JSON.stringify({id:c.id,status:c.status,workbench:c.workbench||null,metadata:c.metadata||{},output:c.enhancedUrl||null},null,2);
  showModal('inspectModal');
}
async function openEditor(id){
  if(DEMO){showToast('Fixture mode is read-only.');return}
  if(batch.busy){showToast('Stop or finish this batch before editing.');return}
  try{
    const body=await json(API.one(id));const c=body.card||body;
    if(['processing','enhancing'].includes(c.status))throw new Error('Card is processing; wait before editing');
    editing={id,revision:c.workbenchRevision||0,rotation:c.workbench?.rotation||0,quad:c.workbench?.quad?c.workbench.quad.map(p=>[...p]):null,local:State.settingsFrom({...settings,...c.workbench?.customSettings})};
    $('editTitle').textContent=c.fileName;
    $('editError').textContent='';
    $('useOverride').checked=Object.keys(c.workbench?.customSettings||{}).length>0;
    $('saveItem').disabled=false;
    $('itemSettings').innerHTML=Object.entries(State.LIMITS).map(([key,[min,max]])=>`<label for="item-${key}">${esc(key)}<output id="item-${key}-value">${editing.local[key]}</output><input id="item-${key}" type="range" min="${min}" max="${max}" step="${key==='scale'?.5:.01}" value="${editing.local[key]}"></label>`).join('')+
      ['chromeOnly','autoCrop','autoRotate'].map(key=>`<label><input id="item-${key}" type="checkbox" ${editing.local[key]?'checked':''}> ${esc(key)}</label>`).join('');
    for(const key of Object.keys(State.DEFAULTS)){
      const input=$('item-'+key);input.disabled=!$('useOverride').checked;
      input.oninput=()=>{editing.local[key]=input.type==='checkbox'?input.checked:Number(input.value);const output=$('item-'+key+'-value');if(output)output.textContent=input.value;};
    }
    editImage=new Image();
    editImage.onload=()=>{if(editing?.id===id)drawCrop()};
    editImage.onerror=()=>{$('editError').textContent='Original image could not be loaded';$('saveItem').disabled=true;};
    editImage.src=c.url;
    showModal('editModal');
  }catch(err){showToast(err.message,4000)}
}
function drawCrop(){
  if(!editing||!editImage?.complete||!editImage.naturalWidth)return;
  const rotated=editing.rotation%180!==0;
  const w=rotated?editImage.naturalHeight:editImage.naturalWidth,h=rotated?editImage.naturalWidth:editImage.naturalHeight;
  const scale=Math.min(1,700/Math.max(w,h));
  const canvas=$('cropCanvas');canvas.width=Math.round(w*scale);canvas.height=Math.round(h*scale);
  const ctx=canvas.getContext('2d');
  ctx.translate(canvas.width/2,canvas.height/2);ctx.rotate(editing.rotation*Math.PI/180);
  ctx.drawImage(editImage,-editImage.naturalWidth*scale/2,-editImage.naturalHeight*scale/2,editImage.naturalWidth*scale,editImage.naturalHeight*scale);
  drawCorners();drawQuadInputs();
}
function drawQuadInputs(){
  const points=editing.quad||[[0,0],[1,0],[1,1],[0,1]];
  $('quadInputs').innerHTML=points.map((p,i)=>`<label>${['TL','TR','BR','BL'][i]} ${p.map((v,k)=>`<input type="number" aria-label="${['TL','TR','BR','BL'][i]} ${k?'y':'x'}" min="0" max="1" step=".001" value="${v.toFixed(3)}" data-point="${i}" data-axis="${k}">`).join(' ')}</label>`).join('');
  $('quadInputs').querySelectorAll('input').forEach(input=>input.onchange=()=>{
    editing.quad=(editing.quad||[[0,0],[1,0],[1,1],[0,1]]).map(p=>[...p]);
    editing.quad[Number(input.dataset.point)][Number(input.dataset.axis)]=clamp01(Number(input.value));drawCorners();
  });
}
function drawCorners(){
  if(!editing)return;
  const points=editing.quad||[[0,0],[1,0],[1,1],[0,1]];
  $('cropOverlay').innerHTML=`<polygon points="${points.map(p=>p.map(v=>v*1000).join(',')).join(' ')}"/>`+points.map((p,i)=>`<circle data-corner="${i}" cx="${p[0]*1000}" cy="${p[1]*1000}" r="18"/>`).join('');
}
let movingCorner=null;
$('cropOverlay').onpointerdown=event=>{
  if(!editing||event.target.dataset.corner===undefined)return;
  movingCorner=Number(event.target.dataset.corner);
  $('cropOverlay').setPointerCapture(event.pointerId);
  editing.quad=(editing.quad||[[0,0],[1,0],[1,1],[0,1]]).map(p=>[...p]);
};
$('cropOverlay').onpointermove=event=>{
  if(movingCorner===null||!editing)return;
  const box=$('cropOverlay').getBoundingClientRect();
  editing.quad[movingCorner]=[clamp01((event.clientX-box.left)/box.width),clamp01((event.clientY-box.top)/box.height)];
  drawCorners();drawQuadInputs();
};
$('cropOverlay').onpointerup=()=>{movingCorner=null};
$('cropOverlay').onpointercancel=()=>{movingCorner=null};
$('rotateCard').onclick=()=>{if(editing){editing.rotation=(editing.rotation+90)%360;editing.quad=null;drawCrop()}};
$('fullFrame').onclick=()=>{if(editing){editing.quad=[[0,0],[1,0],[1,1],[0,1]];drawCrop()}};
$('detectCrop').onclick=()=>{if(editing){editing.quad=null;editing.local.autoCrop=true;$('item-autoCrop').checked=true;drawCrop()}};
$('useOverride').onchange=()=>{for(const input of $('itemSettings').querySelectorAll('input'))input.disabled=!$('useOverride').checked};
$('editForm').onsubmit=async event=>{
  event.preventDefault();if(!editing)return;
  $('saveItem').disabled=true;$('editError').textContent='';
  try{
    const body=await json(`/api/cards/${editing.id}/workbench`,{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify({expectedRevision:editing.revision,workbench:{rotation:editing.rotation,quad:editing.quad,customSettings:$('useOverride').checked?editing.local:{}}})});
    ++loadVersion;cards=cards.map(c=>c.id===body.card.id?body.card:c);natural.delete(body.card.id);
    hideModal('editModal');render();showToast('Item saved; run to create a fresh output.');
  }catch(err){$('editError').textContent=err.message;}
  finally{$('saveItem').disabled=false;}
};
$('editClose').onclick=()=>hideModal('editModal');
$('inspectClose').onclick=()=>hideModal('inspectModal');
for(const id of ['editModal','inspectModal'])$(id).onclick=event=>{if(event.target===$(id))hideModal(id)};
$('addFolder').onclick=()=>{if(!batch.busy)$('folderPicker').click()};
$('folderPicker').onchange=event=>{void uploadFiles(event.target.files);event.target.value='';};
for(const key of ['autoCrop','autoRotate']){
  $(key).checked=settings[key];$(key).onchange=()=>{settings[key]=$(key).checked;saveSettings()};
}
$('resumeBatch').checked=storage.get('cardchop.resume')!==false;
$('resumeBatch').onchange=()=>storage.set('cardchop.resume',$('resumeBatch').checked);
const theme=storage.get('cardchop.theme')==='neon'?'neon':'amber';
$('theme').value=theme;document.documentElement.dataset.theme=theme;
$('theme').onchange=()=>{document.documentElement.dataset.theme=$('theme').value;storage.set('cardchop.theme',$('theme').value)};
