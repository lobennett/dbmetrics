"""Interactive explanation of the existing RGB histogram baseline."""

import base64
from html import escape
from io import BytesIO
import json
from string import Template

from PIL import Image
from dbmetrics.experimental.half_similarity import cosine, histogram


def histogram_data(decoded, bins):
    halves = []
    vectors = []
    for half in decoded["halves"]:
        images = []
        counts = []
        for frame in half["frames"]:
            output = BytesIO()
            Image.frombytes("RGB", (256, 256), frame).save(output, format="JPEG", quality=75)
            images.append("data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode())
            counts.append(histogram([frame], bins))
        vectors.append(histogram(half["frames"], bins))
        halves.append(dict(images=images, counts=counts, times=half["timestamps_seconds"]))
    return dict(bins=bins, halves=halves, score=cosine(*vectors))


def histogram_html(decoded, bins):
    data = json.dumps(histogram_data(decoded, bins)).replace("<", "\\u003c")
    return Template(_PAGE).substitute(data=data)


def show_histograms(decoded, bins):
    from IPython.display import HTML, display

    page = escape(histogram_html(decoded, bins), quote=True)
    display(HTML(f'<iframe title="RGB histogram explorer" srcdoc="{page}" '
                 'style="width:100%;height:1250px;border:0"></iframe>'))


_PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Visual baseline</title><style>
:root { color-scheme:light; --first:#1f77b4; --second:#c86718; }
* {box-sizing:border-box} body {margin:0;padding:24px;font:15px/1.5 system-ui,sans-serif;color:#20252b;background:white}
main {max-width:1050px;margin:auto} h1 {font-size:25px;margin:0 0 8px} h2 {font-size:17px;margin:0 0 8px}
p {margin:8px 0} .muted {color:#5c6670} .panels {display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}
.panel {min-width:0;border:1px solid #dce1e6;border-top:4px solid var(--first);border-radius:6px;padding:16px}
.second {border-top-color:var(--second)} img {display:block;width:100%;max-width:210px;height:auto;margin:10px auto}
svg {width:100%;height:auto;display:block} label {display:block;margin:12px 0 4px;font-weight:600}
input[type=range] {width:100%} select,button,textarea {font:inherit;padding:8px;border:1px solid #bcc5ce;border-radius:5px}
.note,pre {background:#f5f7fa;padding:12px 16px;border-radius:6px} pre {white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}
form {margin-top:20px;border-top:1px solid #dce1e6;padding-top:16px} textarea {width:100%;min-height:70px}
button {background:white;color:#1f77b4;margin-top:8px;cursor:pointer}
@media(max-width:650px) {body{padding:12px}.panels{grid-template-columns:minmax(0,1fr)}}
</style></head><body><main>
<h1>2. Half color similarity</h1>
<p class="muted">Count pixel intensities, accumulate frames, then compare the two half-vectors. No model is involved.</p>
<svg viewBox="0 0 760 64" role="img" aria-label="Frame pixels become RGB counts, then half vectors, then a similarity score">
<title>Pixels → RGB bins → summed half vectors → cosine similarity</title>
<text x="5" y="36">Frame pixels</text><path d="M115 30h35m-8-5 8 5-8 5" stroke="currentColor" fill="none"/>
<text x="165" y="36">RGB bins</text><path d="M255 30h35m-8-5 8 5-8 5" stroke="currentColor" fill="none"/>
<text x="305" y="36">Sum across frames</text><path d="M470 30h35m-8-5 8 5-8 5" stroke="currentColor" fill="none"/>
<text x="520" y="36">Compare half-vectors</text></svg>
<label for="sample">Sample <span id="sample-number"></span></label>
<input id="sample" type="range" min="0" value="0">
<label for="mode">Show counts from</label>
<select id="mode"><option value="frame">This frame only</option><option value="accumulated">All samples up to this frame</option></select>
<p id="scope" class="muted"></p>
<div class="panels">
<section class="panel"><h2>First half</h2><img id="image-0" alt="First-half sampled crop"><p id="time-0"></p><div id="charts-0"></div></section>
<section class="panel second"><h2>Second half</h2><img id="image-1" alt="Second-half sampled crop"><p id="time-1"></p><div id="charts-1"></div></section>
</div>
<p class="muted">X: channel intensity (0–255). Y: pixel counts. Each row uses the same Y-axis for both halves, so bar heights are comparable.</p>
<pre id="formula"></pre>
<div class="note"><strong>Half color similarity, using every sampled frame:</strong> <span id="score"></span><br>
Cosine compares the direction of two concatenated RGB count vectors. This is color similarity, not confidence or a probability that the actions match. The score stays fixed as you inspect frames.</div>
<p class="muted">Counts use the original RGB crops. Preview images are compressed for display only. Repeated sampled frames contribute repeatedly, exactly as in the existing computation.</p>
<form id="quiz"><h2>Explain it back</h2><label for="answer">A pixel has RGB values (20, 180, 240). With 16 bins per channel, which intensity range does it contribute to in each histogram? Does its position in the image matter?</label>
<textarea id="answer" placeholder="Answer here or in our chat. Do not include pixel coordinates in the bin calculation."></textarea>
<button type="submit">Check my explanation</button><p id="status" role="status"></p></form>
</main><script>
const data=$data, byId=id=>document.getElementById(id);
const colors=['#b83e3e','#25814a','#316fb5'], names=['Red','Green','Blue'];
const node=(name,attrs,text)=>{const element=document.createElementNS('http://www.w3.org/2000/svg',name);for(const [key,value] of Object.entries(attrs))element.setAttribute(key,value);if(text!==undefined)element.textContent=text;return element;};
const count=data.halves[0].counts.length;
byId('sample').max=count-1;byId('score').textContent=data.score.toFixed(4);
function render(){
 const sample=Number(byId('sample').value), accumulated=byId('mode').value==='accumulated';
 const frames=accumulated?sample+1:1;
 const vectors=data.halves.map(half=>accumulated?half.counts.slice(0,sample+1).reduce((total,counts)=>total.map((value,index)=>value+counts[index]),Array(3*data.bins).fill(0)):half.counts[sample]);
 byId('sample-number').textContent=(sample+1)+' of '+count;
 byId('scope').textContent=frames+' sample(s) per half · '+(frames*65536).toLocaleString()+' intensity values per channel. The image shows the current frame.';
 data.halves.forEach((half,index)=>{
  byId('image-'+index).src=half.images[sample];byId('time-'+index).textContent='Selected frame at '+half.times[sample].toFixed(3)+' seconds';
  const charts=byId('charts-'+index);charts.replaceChildren();
  for(let channel=0;channel<3;channel++){
   const offset=channel*data.bins, maximum=Math.max(1,...vectors.flatMap(vector=>vector.slice(offset,offset+data.bins)));
   const svg=node('svg',{viewBox:'0 0 400 120',role:'img','aria-label':names[channel]+' pixel intensity histogram'});
   svg.append(node('title',{},names[channel]+' histogram'),node('text',{x:0,y:15,fill:colors[channel]},names[channel]),node('text',{x:398,y:15,'text-anchor':'end','font-size':11},'Y max '+maximum.toLocaleString()));
   svg.append(node('path',{d:'M42 24V96H390',stroke:'#8c959e',fill:'none'}));
   for(let bin=0;bin<data.bins;bin++){
    const value=vectors[index][offset+bin],height=70*value/maximum,width=348/data.bins;
    const bar=node('rect',{x:42+bin*width,y:96-height,width:width-1,height:height,fill:colors[channel]});
    bar.append(node('title',{},Math.ceil(bin*256/data.bins)+'–'+(Math.ceil((bin+1)*256/data.bins)-1)+': '+value.toLocaleString()+' values'));svg.append(bar);
   }
   svg.append(node('text',{x:35,y:99,'text-anchor':'end','font-size':11},'0'),node('text',{x:42,y:114,'font-size':11},'0'),node('text',{x:390,y:114,'text-anchor':'end','font-size':11},'255'));charts.append(svg);
  }
 });
 byId('formula').textContent='Bin index = floor(intensity × '+data.bins+' / 256)\n'+data.bins+' bins × 3 channels = '+(3*data.bins)+' counts in each vector\nFor this display: '+frames+' × 256 × 256 = '+(frames*65536).toLocaleString()+' values per channel\nHalf vector = sum of all '+count+' frame histograms (red bins, then green, then blue).';
}
byId('sample').oninput=render;byId('mode').onchange=render;render();
byId('quiz').onsubmit=event=>{event.preventDefault();const answer=byId('answer').value.trim();if(!answer)return;byId('status').textContent='Copy your answer into our chat so we can discuss it.';};
</script></body></html>'''
