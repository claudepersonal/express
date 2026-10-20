const {test}=require('node:test');
const assert=require('node:assert/strict');
const state=require('../public/workbench-state');
const {validateWorkbench}=require('../workbench-contract');
const {enhanceOptsFrom,optionsForCard}=require('../enhance-contract');

test('one batch owns its snapshot until stop/finish; remaining IDs survive',()=>{
  const run=new state.BatchRun(),settings={...state.DEFAULTS};
  assert.equal(run.start(['first','second'],settings),true);
  settings.sharpen=.9;
  assert.equal(run.settings.sharpen,.5);
  run.complete('first');
  assert.equal(run.start(['third'],settings),false);
  run.stop();assert.equal(run.stopping,true);assert.deepEqual(run.remaining,['second']);
  run.finish();assert.equal(run.busy,false);assert.deepEqual(run.remaining,['second']);
});

test('resume requires this card output, successful status and exact applied geometry/settings',()=>{
  const card={id:'one',fileName:'same.jpg',status:'enhanced',enhancedUrl:'/enhanced/one.png',metadata:{set_name:'Chrome'}};
  const opts=state.effectiveOptions(card,state.DEFAULTS);
  card.metadata.enhancement={options:opts};
  assert.equal(state.resumable(card,opts),true);
  const duplicate={id:'two',fileName:'same.jpg',status:'queued',metadata:{}};
  assert.equal(state.resumable(duplicate,opts),false);
  for(const changed of [{...opts,sharpen:.9},{...opts,rotation:90},{...opts,quad:[[0,0],[1,0],[1,1],[0,1]]}])assert.equal(state.resumable(card,changed),false);
  assert.equal(state.resumable({...card,status:'failed'},opts),false);
  assert.equal(state.resumable({...card,enhancedUrl:null},opts),false);
});

test('per-card overrides and chrome gate use the API units on both client and server',()=>{
  const card={metadata:{set_name:'AEW base'},workbench:{rotation:90,quad:[[0,0],[1,0],[1,1],[0,1]],customSettings:{sharpen:.8,contrast:.12,descratch:.6,chromeOnly:true}}};
  const effective=state.effectiveOptions(card,state.DEFAULTS);
  assert.equal(effective.descratch,0);assert.equal(effective.sharpen,.8);assert.equal(effective.contrast,.12);
  const params=Object.fromEntries(Object.entries(effective).map(([k,v])=>[k,Array.isArray(v)?JSON.stringify(v):String(v)]));
  const parsed=enhanceOptsFrom(params);
  assert.deepEqual(optionsForCard(parsed,card),effective);
  card.metadata.set_name='AEW Chrome';
  assert.equal(state.effectiveOptions(card,state.DEFAULTS).descratch,.6);
  assert.equal(optionsForCard(parsed,card).descratch,.6);
});

test('invalid, crossed, degenerate or out-of-frame crop edits cannot persist',()=>{
  for(const quad of [[],[[0,0],[1,1],[1,0],[0,1]],[[0,0],[0,0],[1,1],[0,1]],[[0,0],[1.1,0],[1,1],[0,1]],[[0,0],[.001,0],[.001,.001],[0,.001]]])assert.throws(()=>validateWorkbench({quad}));
  assert.throws(()=>validateWorkbench({rotation:45}));
  assert.throws(()=>validateWorkbench({customSettings:{contrast:1.2}}));
  assert.throws(()=>validateWorkbench({customSettings:{restorationStrength:.5}}));
  assert.deepEqual(validateWorkbench({quad:[[0,0],[1,0],[1,1],[0,1]]}).quad,[[0,0],[1,0],[1,1],[0,1]]);
});

test('stored settings are validated instead of letting corrupt values reach requests',()=>{
  const s=state.settingsFrom({scale:'4',sharpen:Infinity,contrast:8,chromeOnly:'false'});
  assert.deepEqual(s,state.DEFAULTS);
});
