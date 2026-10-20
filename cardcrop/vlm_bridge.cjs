const vlm=require('./vlm_ocr');
async function main(){
  if(process.argv[2]==='status')return vlm.status();
  if(process.argv[2]!=='read'||!process.argv[3])throw new Error('Expected status or read image options');
  return vlm.vlmReadGated(process.argv[3],JSON.parse(process.argv[4]||'{}'));
}
main().then(result=>{console.log(JSON.stringify(result));if(result.error)process.exitCode=1}).catch(error=>{console.log(JSON.stringify({error:error.message}));process.exitCode=1});
