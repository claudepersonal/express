// Uses the actual server and OpenCV worker. No provider credentials or mocked outputs.
const assert=require('node:assert/strict');
const fs=require('node:fs');
const os=require('node:os');
const path=require('node:path');
const {spawn,execFileSync}=require('node:child_process');
const {once}=require('node:events');
const {chromium}=require('playwright');
const root=path.resolve(__dirname,'..');
const delay=ms=>new Promise(r=>setTimeout(r,ms));
async function waitFor(fn,timeout=60000){
  const start=Date.now();while(Date.now()-start<timeout){const v=await fn();if(v)return v;await delay(150)}
  throw new Error('Timed out waiting for condition');
}
(async()=>{
  const temp=fs.mkdtempSync(path.join(os.tmpdir(),'cardchop-gui-'));
  const image=path.join(temp,'card.png');
  execFileSync('python3',['-c',`from PIL import Image,ImageDraw
im=Image.new('RGB',(200,280),'white');d=ImageDraw.Draw(im)
d.rectangle((5,5,194,274),outline='black',width=3)
d.text((20,70),'2025 AEW WRESTLING',fill='black');d.text((20,150),'KENNY OMEGA',fill='black')
im.save(${JSON.stringify(image)})`]);
  const env={...process.env,PORT:'0',STORAGE_DIR:path.join(temp,'data'),YOLO_MODEL:path.join(temp,'absent.pt')};
  for(const k of ['OPENAI_API_KEY','ANTHROPIC_API_KEY','GOOGLE_API_KEY','GEMINI_API_KEY','HF_TOKEN'])delete env[k];
  const server=spawn('python3',['-m','uvicorn','backend.app:app','--host','0.0.0.0','--port','0'],{cwd:root,env,stdio:['ignore','pipe','pipe']});
  let output='';server.stdout.on('data',b=>output+=b);server.stderr.on('data',b=>output+=b);
  let browser;
  try{
    const port=await waitFor(()=>output.match(/http:\/\/0\.0\.0\.0:(\d+)/)?.[1]);
    const url='http://127.0.0.1:'+port;
    browser=await chromium.launch({headless:true});
    const page=await browser.newPage({viewport:{width:1440,height:1000}});
    const errors=[];page.on('pageerror',e=>errors.push(e.message));
    await page.goto(url);await page.waitForSelector('#emptyAdd');
    const file={name:'duplicate.png',mimeType:'image/png',buffer:fs.readFileSync(image)};
    await page.locator('#filePicker').setInputFiles([file,file]);
    await page.waitForFunction(()=>document.querySelectorAll('.card').length===2);
    const getCards=async()=>(await (await fetch(url+'/api/cards')).json()).cards;
    await waitFor(async()=>(await getCards()).every(c=>!['processing','enhancing'].includes(c.status)));
    await page.locator('details.workbench-options').evaluate(el=>el.open=true);
    await page.locator('#autoRotate').uncheck();
    await page.locator('.card [data-action=edit]').first().click();
    await page.waitForSelector('#editModal.on');
    await page.locator('#rotateCard').click();await page.locator('#fullFrame').click();
    await page.locator('#useOverride').check();await page.locator('#item-autoRotate').uncheck();
    await page.locator('#item-scale').evaluate(el=>{el.value='1';el.dispatchEvent(new Event('input',{bubbles:true}))});
    await page.locator('#saveItem').click();await page.waitForSelector('#editModal.on',{state:'hidden'});
    const edited=(await getCards()).find(c=>c.workbench);
    assert.equal(edited.workbench.rotation,90);assert.equal(edited.enhancedUrl,null);
    const stale=await fetch(url+`/api/cards/${edited.id}/workbench`,{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify({expectedRevision:0,workbench:{rotation:0}})});
    assert.equal(stale.status,409);
    let submissions=0;
    await page.route('**/api/cards/*/enhance?*',async route=>{submissions++;await delay(300);await route.continue()});
    await page.locator('#runSelected').click();await page.locator('#stopBatch').click();
    await page.waitForFunction(()=>document.querySelector('#stopBatch').classList.contains('hidden'),null,{timeout:180000});
    assert.equal(submissions,1,'Stop must prevent submitting the next card');
    await page.locator('#runSelected').click();
    await page.waitForFunction(()=>document.querySelector('#summary').textContent.includes('2 enhanced'),null,{timeout:180000});
    await page.waitForFunction(()=>!document.querySelector('#runSelected').disabled);
    assert.equal(submissions,2,'Resume should skip the already completed card');
    const completed=await getCards();assert.equal(completed.filter(c=>c.status==='enhanced').length,2);
    assert.equal(completed.find(c=>c.id===edited.id).metadata.output_size,'280x200');
    const png=await fetch(url+completed[0].enhancedUrl);assert.equal(png.headers.get('content-type'),'image/png');
    await page.locator('.card').first().dblclick();await page.waitForSelector('#compareModal.on');
    await page.keyboard.press('Escape');await page.waitForSelector('#compareModal.on',{state:'hidden'});
    const downloadPromise=page.waitForEvent('download');await page.locator('#exportAudit').click();
    const download=await downloadPromise;const audit=JSON.parse(fs.readFileSync(await download.path(),'utf8'));
    assert.equal(audit.length,2);assert.ok(audit.every(c=>c.appliedSettings));
    await page.locator('#theme').selectOption('neon');await page.reload();
    await page.waitForFunction(()=>document.querySelectorAll('.card').length===2);
    assert.equal(await page.locator('html').getAttribute('data-theme'),'neon');
    assert.match(await page.locator('#summary').innerText(),/2 selected/);
    await page.setViewportSize({width:390,height:844});
    assert.equal(await page.locator('#runSelected').isVisible(),true);
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),true,'Mobile overflow');
    assert.deepEqual(errors,[]);
    console.log(JSON.stringify({desktop:'passed',mobile:'passed',upload:'real',enhancement:'opencv',stop_resume:'passed',stale_edit:'rejected',duplicates:'independent',audit:'applied settings',errors},null,2));
  }finally{
    await browser?.close();server.kill('SIGTERM');if(server.exitCode===null)await once(server,'exit');
    fs.rmSync(temp,{recursive:true,force:true});
  }
})().catch(err=>{console.error(err);process.exitCode=1});
