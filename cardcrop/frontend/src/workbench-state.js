(function(root,factory){
  const api=factory();
  if(typeof module==='object'&&module.exports)module.exports=api;
  else root.CardChopState=api;
})(typeof globalThis!=='undefined'?globalThis:this,function(){
  const DEFAULTS={scale:2,descratch:.35,denoise:.20,sharpen:.50,contrast:.08,chromeOnly:true,autoCrop:true,autoRotate:true};
  const LIMITS={scale:[1,4],descratch:[0,1],denoise:[0,1],sharpen:[0,1],contrast:[0,.5]};
  function settingsFrom(value){
    const result={...DEFAULTS};
    if(!value||typeof value!=='object')return result;
    for(const [key,range] of Object.entries(LIMITS)){
      const v=value[key];if(typeof v==='number'&&Number.isFinite(v)&&v>=range[0]&&v<=range[1])result[key]=v;
    }
    for(const key of ['chromeOnly','autoCrop','autoRotate'])if(typeof value[key]==='boolean')result[key]=value[key];
    return result;
  }
  function isChrome(card){
    const m=card.metadata||{};
    return /(chrome|foil|refractor|prizm|holo|metallic|dazzler|parallel)/i.test([m.parallel,m.card_type,m.set_name,m.rarity,m.finish,m.foil,m.all_text,m.bio_text].filter(Boolean).join(' '));
  }
  function effectiveOptions(card,global){
    const s=settingsFrom({...global,...card.workbench?.customSettings});
    return {scale:s.scale,descratch:s.chromeOnly&&!isChrome(card)?0:s.descratch,denoise:s.denoise,sharpen:s.sharpen,contrast:s.contrast,
      conservative:false,autoRotate:s.autoRotate,autoCrop:s.autoCrop,rotation:card.workbench?.rotation||0,quad:card.workbench?.quad||null};
  }
  function sameOptions(a,b){
    return !!a&&Object.keys(b).every(k=>JSON.stringify(a[k])===JSON.stringify(b[k]));
  }
  function resumable(card,options){
    return card.status==='enhanced'&&!!card.enhancedUrl&&sameOptions(card.metadata?.enhancement?.options,options);
  }
  class BatchRun {
    constructor(){this.busy=false;this.stopping=false;this.ids=[];this.remaining=[];this.settings=null;}
    start(ids,settings){
      if(this.busy||!ids.length)return false;
      this.busy=true;this.stopping=false;this.ids=[...ids];this.remaining=[...ids];this.settings={...settings};return true;
    }
    complete(id){this.remaining=this.remaining.filter(x=>x!==id);}
    stop(){this.stopping=true;}
    finish(){this.busy=false;this.stopping=false;}
  }
  return {DEFAULTS,LIMITS,settingsFrom,isChrome,effectiveOptions,sameOptions,resumable,BatchRun};
});
