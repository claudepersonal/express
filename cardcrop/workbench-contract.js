// Normalized coordinates match the handoff CropQuad (TL, TR, BR, BL).
const LIMITS = {scale:[1,4],descratch:[0,1],denoise:[0,1],sharpen:[0,1],contrast:[0,.5]};
const FULL_FRAME = [[0,0],[1,0],[1,1],[0,1]];
function validateQuad(quad) {
  if (quad === null) return null;
  if (!Array.isArray(quad) || quad.length !== 4 || quad.some(p => !Array.isArray(p) || p.length !== 2 || p.some(v => typeof v !== 'number' || !Number.isFinite(v) || v < 0 || v > 1))) throw new Error('Crop requires four normalized corners');
  const crosses = quad.map((p,i) => {
    const b=quad[(i+1)%4],c=quad[(i+2)%4];
    return (b[0]-p[0])*(c[1]-b[1])-(b[1]-p[1])*(c[0]-b[0]);
  });
  const area=quad.reduce((a,p,i)=>{const b=quad[(i+1)%4];return a+p[0]*b[1]-b[0]*p[1];},0)/2;
  if (crosses.some(v=>v<=0) || area < .005) throw new Error('Crop corners must form a clockwise convex card with nonzero area');
  return quad.map(p=>[...p]);
}
function validateWorkbench(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Invalid workbench');
  if (Object.keys(value).some(k=>!['rotation','quad','customSettings'].includes(k))) throw new Error('Unknown workbench field');
  const rotation=value.rotation??0;
  if (!Number.isInteger(rotation) || ![0,90,180,270].includes(rotation)) throw new Error('Rotation must be 0, 90, 180 or 270');
  const customSettings=value.customSettings??{};
  if (!customSettings || typeof customSettings !== 'object' || Array.isArray(customSettings)) throw new Error('Invalid custom settings');
  for (const [k,v] of Object.entries(customSettings)) {
    if (LIMITS[k]) {
      if (typeof v !== 'number' || !Number.isFinite(v) || v<LIMITS[k][0] || v>LIMITS[k][1]) throw new Error(`Invalid ${k}`);
    } else if (['chromeOnly','autoCrop','autoRotate'].includes(k)) {
      if (typeof v !== 'boolean') throw new Error(`Invalid ${k}`);
    } else throw new Error(`Unsupported setting: ${k}`);
  }
  return {rotation,quad:validateQuad(value.quad??null),customSettings:{...customSettings}};
}
module.exports={LIMITS,FULL_FRAME,validateQuad,validateWorkbench};
