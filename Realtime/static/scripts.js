function pauseDetection(){
  fetch('/api/stop',{method:'POST'});
}

function playAlert(){
  const ctx = new AudioContext();
  const osc = ctx.createOscillator();
  osc.frequency.value = 880;
  osc.connect(ctx.destination);
  osc.start();
  osc.stop(ctx.currentTime + 0.3);
}

setInterval(()=>{
  fetch('/api/detections')
    .then(r=>r.json())
    .then(d=>{
      if(d.some(x=>x.type==="weapon")){
        playAlert();
      }
    });
},3000);
