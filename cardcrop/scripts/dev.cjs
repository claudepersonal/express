const {spawn}=require('node:child_process');
const path=require('node:path');
const root=path.resolve(__dirname,'..');
const children=[spawn('python3',['-m','uvicorn','backend.app:app','--host','127.0.0.1','--port','8000','--reload'],{cwd:root,stdio:'inherit'}),spawn('npm',['run','dev'],{cwd:path.join(root,'frontend'),stdio:'inherit'})];
let stopping=false;
function stop(){if(stopping)return;stopping=true;children.forEach(child=>child.kill('SIGTERM'))}
for(const event of ['SIGTERM','SIGINT'])process.on(event,stop);
for(const child of children)child.on('exit',code=>{stop();process.exitCode=code||0});
