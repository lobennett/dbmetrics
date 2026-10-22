"""Notebook-only view of the frames already selected by decode_media."""

import base64
from io import BytesIO
import json
from string import Template

from PIL import Image


def show_split_samples(decoded):
    from html import escape
    from IPython.display import HTML, display

    page = escape(split_sample_html(decoded), quote=True)
    display(HTML(f'<iframe title="Split and sample explorer" srcdoc="{page}" '
                 'style="width:100%;height:1150px;border:0"></iframe>'))


def split_sample_html(decoded):
    halves = []
    for half in decoded["halves"]:
        images = []
        for frame in half["frames"]:
            image = Image.frombytes("RGB", (256, 256), frame)
            output = BytesIO()
            image.save(output, format="JPEG", quality=75)
            images.append("data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode())
        halves.append({
            "images": images,
            "requested": half["requested_seconds"],
            "actual": half["timestamps_seconds"],
            "indices": half["frame_indices"],
            "distinct": len(set(half["frame_indices"])),
        })
    data = json.dumps({"duration": decoded["duration_seconds"],
                       "split": decoded["split_seconds"], "halves": halves}).replace("<", "\\u003c")
    return Template(_PAGE).substitute(data=data)


_PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Split and sample</title>
<style>
:root { color-scheme:light; --first:#1f77b4; --second:#c86718; }
* { box-sizing:border-box; } body { margin:0; padding:24px; font:15px/1.5 system-ui,sans-serif; color:#20252b; background:#fff; }
main { max-width:1000px; margin:auto; } h1 { margin:0 0 8px; font-size:25px; } h2 { font-size:17px; margin:0 0 10px; }
p { margin:8px 0; } .muted { color:#5c6670; } .panels { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:20px; }
.panel { min-width:0; border:1px solid #dce1e6; border-top:4px solid var(--first); border-radius:6px; padding:16px; }
.panel.second { border-top-color:var(--second); } img { display:block; width:100%; max-width:256px; height:auto; margin:12px auto; }
svg { width:100%; height:auto; } label { display:block; margin:18px 0 8px; font-weight:600; } input[type=range] { width:100%; }
dl { margin:0; } dl div { display:flex; flex-wrap:wrap; justify-content:space-between; gap:8px; } dt { color:#5c6670; } dd { margin:0; font-variant-numeric:tabular-nums; }
.note { padding:12px 16px; background:#f5f7fa; border-radius:6px; margin:16px 0; }
form { margin-top:22px; border-top:1px solid #dce1e6; padding-top:18px; } textarea { width:100%; min-height:85px; font:inherit; padding:10px; border:1px solid #bcc5ce; border-radius:5px; }
button { font:inherit; padding:8px 14px; margin-top:8px; border:1px solid #1f77b4; border-radius:5px; background:#fff; color:#1f77b4; cursor:pointer; }
pre { white-space:pre-wrap; overflow-wrap:anywhere; padding:12px; background:#f5f7fa; font-size:13px; }
@media(max-width:600px) { body { padding:14px; } .panels { grid-template-columns:minmax(0,1fr); } }
</style></head><body><main>
<h1>1. Split and sample</h1>
<p class="muted">One real video → two time intervals → two stacks of selected images. No similarity score yet.</p>
<svg id="timeline" viewBox="0 0 900 120" role="img" aria-label="Video timeline split into two halves">
<title>Requested sampling times and actual selected frames</title>
<rect x="30" y="28" width="420" height="30" fill="#e5f0f8"/><rect x="450" y="28" width="420" height="30" fill="#fff0e3"/>
<line x1="450" y1="20" x2="450" y2="75" stroke="#20252b" stroke-width="2"/>
<text x="30" y="98">0 seconds</text><text id="split" x="450" y="98" text-anchor="middle"></text><text id="end" x="870" y="98" text-anchor="end"></text>
<g id="ticks"></g><g id="chosen"></g></svg>
<label for="sample">Inspect sample <span id="sample-number"></span> in each half</label>
<input id="sample" type="range" min="0" value="0" aria-label="Selected sample number">
<p class="muted">Move the slider to inspect existing samples. It does not change sampling or rerun the model.</p>
<div class="panels">
<section class="panel"><h2>First half</h2><p id="bounds-0"></p><img id="image-0" alt="Selected comparison crop from the first half"><dl id="facts-0"></dl></section>
<section class="panel second"><h2>Second half</h2><p id="bounds-1"></p><img id="image-1" alt="Selected comparison crop from the second half"><dl id="facts-1"></dl></section>
</div>
<div class="note"><strong>What you are seeing:</strong> 256 × 256 RGB comparison crops—not the whole original frame. Resizing preserves aspect ratio, then a center crop removes the edges. Frame numbers below start at zero.</div>
<pre id="formula"></pre>
<p class="muted">Each target is the midpoint of an equal time bin. Select the nearest frame inside that half; ties choose the earlier frame. A frame exactly at the split belongs to the second half.</p>
<form id="quiz" data-lavish-question="sampling-understanding"><h2>Explain it back</h2>
<label for="answer">If a half has 300 original frames and we request 64 samples, are we examining every frame? What would “64 samples, 20 distinct frames” mean?</label>
<textarea id="answer" required placeholder="Explain in your own words."></textarea>
<button type="submit">Send my explanation</button><p id="quiz-status" role="status"></p></form>
</main><script>
const data=$data;
const byId=id=>document.getElementById(id), seconds=value=>value.toFixed(3)+' s';
const count=data.halves[0].images.length;
byId('sample').max=count-1;
byId('split').textContent=seconds(data.split)+' · split'; byId('end').textContent=seconds(data.duration);
const svgNode=(name, attrs)=>{const node=document.createElementNS('http://www.w3.org/2000/svg',name); for(const [key,value] of Object.entries(attrs)) node.setAttribute(key,value); return node;};
const position=time=>30+840*time/data.duration;
data.halves.forEach((half,index)=>{half.requested.forEach(time=>byId('ticks').append(svgNode('line',{x1:position(time),x2:position(time),y1:28,y2:58,stroke:index?'#c86718':'#1f77b4','stroke-opacity':0.35})));});
function render(){
 const sample=Number(byId('sample').value); byId('sample-number').textContent=(sample+1)+' of '+count;
 byId('chosen').replaceChildren();
 data.halves.forEach((half,index)=>{
  const start=index?data.split:0, end=index?data.duration:data.split;
  byId('bounds-'+index).textContent=seconds(start)+' ≤ time < '+seconds(end);
  byId('image-'+index).src=half.images[sample];
  const facts=byId('facts-'+index); facts.replaceChildren();
  for(const [label,value] of [['Requested time',seconds(half.requested[sample])],['Selected frame time',seconds(half.actual[sample])],['Original frame number',half.indices[sample]],['Samples / distinct frames',half.images.length+' / '+half.distinct]]){
   const row=document.createElement('div'), term=document.createElement('dt'), description=document.createElement('dd'); term.textContent=label; description.textContent=value; row.append(term,description); facts.append(row);
  }
  byId('chosen').append(svgNode('line',{x1:position(half.requested[sample]),x2:position(half.requested[sample]),y1:18,y2:70,stroke:'#20252b','stroke-dasharray':'3 2'}),svgNode('circle',{cx:position(half.actual[sample]),cy:43,r:6,fill:index?'#c86718':'#1f77b4'}));
 });
 byId('formula').textContent='Requested time = half start + (sample index + 0.5) × half duration / '+count+'\nFirst-half sample '+(sample+1)+': 0 + ('+sample+' + 0.5) × '+data.split.toFixed(3)+' / '+count+' = '+seconds(data.halves[0].requested[sample])+'\nDashed line = requested time; colored dot = selected frame time.';
}
byId('sample').oninput=render; render();
byId('quiz').onsubmit=event=>{event.preventDefault(); const answer=byId('answer').value.trim(); if(!answer)return;
 if(window.lavish){window.lavish.queuePrompt('Check my understanding of split-and-sample: '+answer,{tag:'quiz',queueKey:'sampling-understanding',data:{question:'sampling-understanding',answer}}); window.lavish.sendQueuedPrompts(); byId('quiz-status').textContent='Explanation sent.';}
 else byId('quiz-status').textContent='Copy your explanation into our chat so we can discuss it before moving on.';
};
</script></body></html>'''
