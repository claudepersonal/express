const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const os=require('node:os');
const path=require('node:path');
const {spawn,execFileSync}=require('node:child_process');
const {once}=require('node:events');

test('upload, OCR, validated enhancement and PNG delivery', {timeout:30000}, async()=>{
  const dir=fs.mkdtempSync(path.join(os.tmpdir(),'wrestling-http-'));
  const fixture=path.join(dir,'fixture.png');
  execFileSync('python3',['-c',`import cv2,numpy as np
im=np.full((350,500,3),220,np.uint8)
cv2.putText(im,'AEW WRESTLING',(25,100),cv2.FONT_HERSHEY_SIMPLEX,1,(20,20,20),2)
cv2.putText(im,'KENNY OMEGA',(25,170),cv2.FONT_HERSHEY_SIMPLEX,1,(20,20,20),2)
cv2.imwrite(${JSON.stringify(fixture)},im)`]);
  const server=spawn(process.execPath,[path.resolve(__dirname,'../server.js')],{
    env:{...process.env,PORT:'0',STORAGE_DIR:path.join(dir,'data'),YOLO_MODEL:path.join(dir,'absent.pt')},stdio:['ignore','pipe','pipe']});
  let logs='';server.stdout.on('data',b=>logs+=b);server.stderr.on('data',b=>logs+=b);
  const waitFor=async(fn)=>{for(let i=0;i<100;i++){const v=await fn();if(v)return v;await new Promise(r=>setTimeout(r,100));}throw new Error('Timed out: '+logs);};
  try{
    const port=await waitFor(()=>logs.match(/http:\/\/0\.0\.0\.0:(\d+)/)?.[1]);
    const root='http://127.0.0.1:'+port;
    const get=async url=>(await fetch(root+url)).json();
    const h=await waitFor(async()=>{const h=await get('/health');return h.enhancement.available&&h;});
    assert.equal(h.category,'wrestling');assert.equal(h.yolo.available,false);
    const form=new FormData();form.append('files',new Blob([fs.readFileSync(fixture)],{type:'image/png'}),'fixture.png');
    const up=await (await fetch(root+'/api/cards/upload',{method:'POST',body:form})).json();
    const id=up.cards[0].id;
    const poll=()=>waitFor(async()=>{const {card}=await get('/api/cards/'+id);return !['processing','enhancing'].includes(card.status)&&card;});
    let card=await poll();assert.equal(card.status,'ocr_complete');assert.equal(card.metadata.promotion,'AEW');assert.equal(card.metadata.review_needed,true);
    assert.equal((await fetch(root+'/api/cards/'+id+'/enhance?scale=NaN',{method:'POST'})).status,400);
    assert.equal((await fetch(root+'/api/cards/'+id+'/enhance?scale=2&autoRotate=false&sharpen=0',{method:'POST'})).status,200);
    card=await poll();assert.equal(card.status,'enhanced',JSON.stringify(card));assert.equal(card.metadata.output_size,'1000x700');
    assert.equal(card.metadata.enhancement.detection_mode,'whole_frame');
    const out=await fetch(root+card.enhancedUrl);assert.equal(out.status,200);assert.equal(out.headers.get('content-type'),'image/png');
    assert.equal((await fetch(root+'/api/cards/'+id+'/enhance?cleanup=classical',{method:'POST'})).status,400);
    assert.equal((await fetch(root+'/api/cards/'+id+'/enhance?cleanup=classical&conservative=false&detection=none&scale=1&autoRotate=false',{method:'POST'})).status,200);
    card=await poll();assert.equal(card.status,'enhanced',JSON.stringify(card));
    assert.equal(card.metadata.enhancement.cleanup_engine,'classical');
    assert.equal(card.metadata.enhancement.clahe.clip_limit,1.5);
    assert.equal(card.metadata.output_size,'500x350');
    // Broken input must transition to failed, never enhanced, even with old output.
    const uploaded=fs.readdirSync(path.join(dir,'data/uploads'))[0];
    fs.writeFileSync(path.join(dir,'data/uploads',uploaded),'broken image');
    await fetch(root+'/api/cards/'+id+'/enhance',{method:'POST'});
    card=await poll();assert.equal(card.status,'failed');assert.ok(card.metadata.enhanceError);
  }finally{
    server.kill('SIGTERM');await once(server,'exit');fs.rmSync(dir,{recursive:true,force:true});
  }
});
