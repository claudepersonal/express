const {test}=require('node:test');
const assert=require('node:assert/strict');
const {enhanceOptsFrom}=require('../enhance-contract');

test('cleanup and crop options cross the API boundary',()=>{
  const opts=enhanceOptsFrom({cleanup:'classical',detection:'contour',conservative:'false',cleanupStrength:'0.45'});
  assert.equal(opts.cleanup,'classical');
  assert.equal(opts.detection,'contour');
  assert.equal(opts.cleanupStrength,0.45);
  assert.equal(enhanceOptsFrom({}).cleanup,'manual');
});

test('invalid modes and conservative profile conflicts fail',()=>{
  for(const query of [{cleanup:'unknown'},{detection:['contour']},{cleanupStrength:'NaN'},
    {cleanupStrength:'2'},{cleanup:'cnn'},{cleanup:'classical',conservative:'true'}]){
    assert.throws(()=>enhanceOptsFrom(query));
  }
});
