"use strict";
// FlashHead port of MuseTalk's WebRTC latency wall (templates/webrtc_wall.py).
// Each tile is an iframe running /webrtc/player/{call id}; the page drives the
// server-side group and never touches media itself.
const initialMatch=location.pathname.match(/^\/webrtc\/groups\/([^/]+)\/wall$/);
const debugModes=["off","docked","overlay"];
const SPEAKING=["preparing","armed","speaking","draining"];
const ENDED=["missing","closed","error"];
let groupId=initialMatch?decodeURIComponent(initialMatch[1]):null;
let debugMode=readPref("webrtcWallDebugMode","off");
if(!debugModes.includes(debugMode)) debugMode="off";
let currentGroup=null;
let audioLeader=null;
let lastTts=null;
let health=null;
let kokoro=null;
let gpu={};
let pendingRetry=null;
let busy=false;
let polling=false;
const clientStats=new Map();
const sendErrors=new Map();
const $=id=>document.getElementById(id);

function readPref(key,fallback) { try{return localStorage.getItem(key)||fallback}catch(_){return fallback} }
function writePref(key,value) { try{localStorage.setItem(key,value)}catch(_){} }
function setStatus(message,kind="") { const el=$("status"); el.textContent=message; el.className=`status ${kind}`; }
function finiteNumber(value) { if(value===null||value===undefined||value==="")return null;const n=Number(value);return Number.isFinite(n)?n:null; }
function format(value,digits=1,suffix="") { const n=finiteNumber(value); return n!==null?`${n.toFixed(digits)}${suffix}`:"n/a"; }
function plural(n,word) { return `${n} ${word}${n===1?"":"s"}`; }

// The API token lives in this page's memory only; player frames ask for it.
function token() { return $("token").value.trim(); }
function showToken(show) { $("tokenRow").hidden=!show; }
async function send(url,options={}) {
  const headers={...(options.headers||{})}; const credential=token(); if(credential) headers.Authorization=`Bearer ${credential}`;
  const response=await fetch(url,{...options,headers,cache:"no-store"});
  if(response.status===401) showToken(true);
  if(!response.ok){const text=(await response.text()).slice(0,600);const error=new Error(response.status===401?"API token required or rejected. Enter it above, then Refresh.":`${response.status} ${text}`.trim());error.status=response.status;throw error}
  return response;
}
async function request(url,options={}) { const response=await send(url,options); const text=await response.text(); let body=text; try{body=text?JSON.parse(text):{}}catch(_){} return {body,response}; }

function sessions() { return currentGroup?.sessions||[]; }
function playerUrl(session) { const url=new URL(session.player_url,location.origin); url.searchParams.set("debug",debugMode); url.searchParams.set("muted",session.session_id===audioLeader?"0":"1"); return url.pathname+url.search; }
function postToFrame(sessionId,message) { const frame=document.querySelector(`iframe[data-session-id="${CSS.escape(sessionId)}"]`); try{frame?.contentWindow?.postMessage(message,location.origin)}catch(_){} }
function applyAudioLeader(sessionId) { audioLeader=sessionId; document.querySelectorAll(".card[data-session-id]").forEach(card=>{ const audible=card.dataset.sessionId===audioLeader; card.classList.toggle("audible",audible); const button=card.querySelector('[data-role="audio"]'); if(button) button.textContent=audible?"Audio on":"Use audio"; postToFrame(card.dataset.sessionId,{type:"webrtc-audio",muted:!audible}); }); }

function tileStats(session) {
  const s=session.stats||{}; const turn=session.last_turn; const client=clientStats.get(session.session_id)||{};
  return [
    `turn ${turn?turn.status:"none"}`,
    `queue ${s.queued_turns??0} turns · ${s.queued_chunks??0} chunks`,
    `first media ${format(turn?.first_media_s,2,"s")}`,
    `audio hold ${format((s.audio_hold_samples??0)/48000,2,"s")}`,
    `neural ${s.generated_frames??0} · idle ${s.idle_frames??0}`,
    `held ${s.held_frames??0} · underrun ${s.underrun_frames??0}`,
    `client JB Δ ${format(client.avJitterBufferDeltaMs,1,"ms")}`,
    `fps ${format(client.videoFps,1)} · drops ${client.droppedFrames??0}`,
    `jitter A/V ${format(client.audioJitterMs,1,"ms")}/${format(client.videoJitterMs,1,"ms")}`,
    `${client.videoCodec||"video"}/${client.audioCodec||"audio"} · RTT ${format(client.rttMs,1,"ms")}`
  ];
}
function tileState(session) { const client=clientStats.get(session.session_id); if(ENDED.includes(session.status)) return session.status; if(client?.pc&&!["connected","idle"].includes(client.pc)&&!SPEAKING.includes(session.status)) return client.pc; return session.status||"created"; }

function buildCard(session) {
  const card=document.createElement("article"); card.className="card"; card.dataset.sessionId=session.session_id;
  const head=document.createElement("div"); head.className="card-head";
  const title=document.createElement("span"); title.className="card-title"; title.textContent=`#${session.index+1} · seed ${session.seed} · ${session.session_id}`; title.title=session.session_id;
  const state=document.createElement("span"); state.className="pill"; state.dataset.role="state";
  const speak=document.createElement("button"); speak.dataset.role="speak"; speak.textContent="Speak"; speak.title="Kokoro text to this tile only"; speak.addEventListener("click",()=>run(()=>startKokoro([session.session_id])));
  const stop=document.createElement("button"); stop.dataset.role="interrupt"; stop.textContent="Stop"; stop.title="Interrupt this tile's speech; the call stays connected"; stop.addEventListener("click",()=>run(()=>interruptPeers([session.session_id])));
  const audio=document.createElement("button"); audio.dataset.role="audio"; audio.addEventListener("click",()=>applyAudioLeader(session.session_id));
  head.append(title,state,speak,stop,audio);
  const frame=document.createElement("iframe"); frame.dataset.sessionId=session.session_id; frame.allow="autoplay; fullscreen"; frame.loading="eager"; frame.title=`Peer ${session.index+1} video`; frame.src=playerUrl(session);
  const stats=document.createElement("div"); stats.className="card-stats"; stats.dataset.role="stats";
  const error=document.createElement("div"); error.className="card-error"; error.dataset.role="error"; error.hidden=true;
  card.append(head,frame,stats,error); return card;
}
function updateCard(card,session) {
  card.querySelector('[data-role="state"]').textContent=tileState(session);
  const stats=card.querySelector('[data-role="stats"]'); stats.replaceChildren(...tileStats(session).map(value=>{const span=document.createElement("span");span.textContent=value;span.title=value;return span;}));
  const message=sendErrors.get(session.session_id)||(session.error?`Server: ${session.error}`:"")||(ENDED.includes(session.status)?"Call ended. Reconnect peers replaces it with a new call (same seed).":"");
  const error=card.querySelector('[data-role="error"]'); error.textContent=message; error.hidden=!message;
  const ended=ENDED.includes(session.status);
  card.querySelector('[data-role="speak"]').disabled=busy||ended||kokoro?.available===false;
  card.querySelector('[data-role="interrupt"]').disabled=busy||ended;
  const frame=card.querySelector("iframe"); const current=new URL(frame.getAttribute("src")||"/",location.origin); const wanted=new URL(session.player_url,location.origin); if(current.pathname!==wanted.pathname) frame.src=playerUrl(session);
}
function renderWall() {
  const list=sessions(); const wall=$("wall"); if(!list.length){wall.className="empty";wall.replaceChildren();wall.textContent=groupId?"Loading group…":"Create a group to populate the wall.";return}
  if(!audioLeader||!list.some(item=>item.session_id===audioLeader)) audioLeader=list[0].session_id;
  wall.className="wall"; if(!wall.querySelector(".card")) wall.textContent="";
  const existing=new Map([...wall.querySelectorAll(".card")].map(card=>[card.dataset.sessionId,card])); const next=[];
  for(const session of list){const card=existing.get(session.session_id)||buildCard(session);existing.delete(session.session_id);updateCard(card,session);next.push(card)}
  // Do not replace existing iframe nodes during the two-second stats refresh.
  // Detaching a connected iframe closes its RTCPeerConnection, and the server
  // releases a call whose peer closes, turning a healthy stream into a dead tile.
  for(const stale of existing.values()) stale.remove();
  next.forEach((card,index)=>{if(wall.children[index]!==card)wall.insertBefore(card,wall.children[index]||null)});
  applyAudioLeader(audioLeader);
}

function renderEndpoints() {
  const box=$("endpoints"); if(!currentGroup){box.hidden=true;return}
  const c=currentGroup.config; box.hidden=false;
  box.textContent=`Group ${currentGroup.group_id} · ${currentGroup.avatar_name} (${currentGroup.avatar_id}) · seeds ${c.first_seed}–${c.first_seed+c.count-1}\nWall ${location.origin}${currentGroup.wall_url}\nRender ${c.width}×${c.height} · ${c.steps} steps · ${c.fps}fps · batch ${c.batch_size} · up to ${c.max_active_calls} rendering call${c.max_active_calls===1?"":"s"} · idle ${c.idle_policy} · ICE ${c.ice_transport_policy}`;
}

function applyProfile() {
  if(!health) return;
  $("renderFps").value=`${health.fps}`; $("resolution").value=`${health.width}×${health.height}`; $("batchSize").value=`${health.batch}`; $("activeCalls").value=`${health.max_active_calls} / ${health.max_sessions} peers`;
  $("count").max=String(Math.min(12,health.max_sessions));
  const aspect=health.width/health.height; document.documentElement.style.setProperty("--tile-aspect",`${health.width}/${health.height}`); // Portrait tiles stop growing at a readable size instead of filling the row.
  document.documentElement.style.setProperty("--tile-min",aspect<1?"250px":"310px"); document.documentElement.style.setProperty("--tile-max",aspect<1?"380px":"1fr");
  const flags=["optimized","real_rope","int8_weights","compile_color","compile_audio","trt_ffn","trt_vae"].filter(key=>health[key]).map(key=>key.replace("_","-"));
  $("profile").textContent=`${health.width} × ${health.height} · ${health.steps} steps · ${health.fps} fps · batch ${health.batch} · memory ${health.memory_mode}${flags.length?` · ${flags.join(", ")}`:""}\n`+
    `Up to ${health.max_active_calls} rendering call${health.max_active_calls===1?"":"s"} / ${health.max_sessions} connected peers · ${health.calls} call${health.calls===1?"":"s"} open · idle ${health.idle_policy}${health.idle_asset?.file?` (${health.idle_asset.file})`:""} · H264 ${health.h264_preset}\n`+
    `${health.ready?"GPU ready":"GPU not ready"} · FPS, batch and resolution are server launch options; restart the server to change them.`;
}
function renderAvatars(body) {
  const select=$("avatarId"); const previous=currentGroup?.avatar_id||select.value; const avatars=body.avatars||[];
  select.replaceChildren(...avatars.map(avatar=>new Option(avatar.kind==="image"?`${avatar.name} · portrait`:avatar.name,avatar.id)));
  select.value=avatars.some(avatar=>avatar.id===previous)?previous:(body.default||avatars[0]?.id||"default");
}
function renderVoices() {
  if(!kokoro) return; const select=$("ttsVoice"); const previous=select.value;
  select.replaceChildren(...(kokoro.voices||[]).map(voice=>new Option(voice,voice)));
  if((kokoro.voices||[]).includes(previous)) select.value=previous;
}
async function refreshServer() {
  const [h,k,a,c]=await Promise.allSettled([request("/health"),request("/webrtc/tts/kokoro/status"),request("/avatars"),request("/config")]);
  if(c.status==="fulfilled") showToken(Boolean(c.value.body.authRequired));
  if(a.status==="fulfilled") renderAvatars(a.value.body);
  if(h.status==="fulfilled"){health=h.value.body;applyProfile()}else{health=null;$("profile").textContent=h.reason.message}
  if(k.status==="fulfilled"){kokoro=k.value.body;renderVoices()}else kokoro=null;
  syncControls(); await refreshMetrics();
  if(h.status==="rejected") throw h.reason;
  return health;
}

function metric(label,value,sub) { const card=document.createElement("div");card.className="metric";const small=document.createElement("small");small.textContent=label;const strong=document.createElement("strong");strong.textContent=value;const span=document.createElement("span");span.textContent=sub;card.append(small,strong,span);return card; }
function renderRate() {
  const chunks=(health?.recent_chunks||[]).filter(chunk=>finiteNumber(chunk.wall_s)>0&&finiteNumber(chunk.frames)>0);
  if(!chunks.length) return null;
  const frames=chunks.reduce((n,chunk)=>n+chunk.frames,0); const seconds=chunks.reduce((n,chunk)=>n+chunk.wall_s,0);
  return {fps:frames/seconds,batch:chunks.reduce((n,chunk)=>n+(chunk.batch||1),0)/chunks.length};
}
function kokoroSummary() {
  if(lastTts) return [format(lastTts.ms,0,"ms"),`${format(lastTts.audio,2,"s")} audio · RTF ${format(lastTts.rtf,2)}${lastTts.cold?" · cold":""}`];
  if(!kokoro) return ["n/a","status unavailable"];
  if(!kokoro.available) return ["missing","install requirements-tts.txt on the server"];
  return [kokoro.busy?"busy":"ready","local CPU; first call is cold"];
}
async function refreshMetrics() {
  const [h,g]=await Promise.allSettled([request("/health"),request("/stats/gpu-live")]);
  if(h.status==="fulfilled"){health=h.value.body;applyProfile()} if(g.status==="fulfilled") gpu=g.value.body;
  renderMetrics();
}
// Pure render from current state; every group update calls it so tiles and metrics agree.
function renderMetrics() {
  const list=sessions(); const clients=list.map(s=>clientStats.get(s.session_id)).filter(Boolean);
  const firstMedia=list.map(s=>finiteNumber(s.last_turn?.first_media_s)).filter(v=>v!==null);
  const speaking=list.filter(s=>SPEAKING.includes(s.status)).length; const queued=list.reduce((n,s)=>n+(s.stats?.queued_turns||0),0);
  const jb=clients.map(c=>finiteNumber(c.avJitterBufferDeltaMs)).filter(v=>v!==null).map(Math.abs);
  const underruns=list.reduce((n,s)=>n+(s.stats?.underrun_frames||0),0); const holds=list.map(s=>(s.stats?.audio_hold_samples||0)/48000);
  const rendered=renderRate(); const [ttsValue,ttsSub]=kokoroSummary();
  $("metrics").replaceChildren(
    metric("Peers",String(list.length),`${list.filter(s=>s.connected).length} connected`),
    metric("Speaking",String(speaking),`${plural(queued,"queued turn")}`),
    metric("First media",firstMedia.length?format(Math.max(...firstMedia)*1000,0,"ms"):"n/a","worst peer; submit → first sent frame"),
    metric("Neural render",rendered?format(rendered.fps,1," fps"):"n/a",health?`GPU while rendering · target ${health.fps}/speaker`:"GPU frames per second"),
    metric("Client buffer Δ",jb.length?format(Math.max(...jb),1,"ms"):"n/a","audio vs video jitter buffer"),
    metric("Underruns",String(underruns),`speech frames held · worst hold ${format(holds.length?Math.max(...holds):0,2,"s")}`),
    metric("GPU",gpu.available===false?"n/a":format(gpu.gpu_util_pct,0,"%"),gpu.available===false?(gpu.reason||"nvidia-smi unavailable"):`${format(gpu.memory_used_gb,1,"GB")} VRAM`),
    metric("Kokoro",ttsValue,ttsSub)
  );
}

function syncControls() {
  const hasGroup=Boolean(currentGroup);
  $("createBtn").disabled=busy; $("refreshBtn").disabled=busy; $("uploadBtn").disabled=busy;
  $("reconnectBtn").disabled=busy||!hasGroup; $("interruptBtn").disabled=busy||!hasGroup; $("deleteBtn").disabled=busy||!hasGroup;
  $("kokoroBtn").disabled=busy||kokoro?.available===false; $("retryBtn").disabled=busy||!pendingRetry;
  $("retryBtn").textContent=pendingRetry?`Retry ${plural(pendingRetry.sessionIds?.length||sessions().length,"failed peer")}`:"Retry failed peers";
  document.querySelectorAll(".card[data-session-id]").forEach(card=>{const session=sessions().find(s=>s.session_id===card.dataset.sessionId);if(session)updateCard(card,session)});
}
async function run(action) { if(busy)return; busy=true; syncControls(); try{await action()}catch(error){setStatus(error.message,"err")}finally{busy=false;syncControls()} }

function adoptGroup(group) {
  if(!currentGroup){
    // First load of a shared group URL: show the settings it was created with.
    if([...$("avatarId").options].some(option=>option.value===group.avatar_id)) $("avatarId").value=group.avatar_id;
    $("count").value=String(group.config.count); $("seed").value=String(group.config.first_seed);
  }
  currentGroup=group; groupId=group.group_id;
  if(initialMatch&&location.pathname!==group.wall_url) history.replaceState(null,"",group.wall_url);
  renderEndpoints(); renderWall(); syncControls(); renderMetrics();
}
async function createGroup() {
  if(!health?.ready){await refreshServer();if(!health?.ready)throw new Error("GPU server is not ready.")}
  if(currentGroup){setStatus("Releasing the previous group…","warn");await deleteGroup(true)}
  setStatus("Creating FlashHead calls…","warn"); const params=new URLSearchParams({count:$("count").value,seed:$("seed").value,avatar_id:$("avatarId").value});
  const group=(await request(`/webrtc/groups/create?${params}`,{method:"POST"})).body;
  audioLeader=null; pendingRetry=null; sendErrors.clear(); clientStats.clear(); adoptGroup(group); await refreshMetrics();
  setStatus(`Created ${plural(group.sessions.length,"peer")}. Audio is enabled only on tile 1.`,"ok");
}
async function refreshGroup(silent=false) {
  const id=currentGroup?.group_id||groupId; if(!id)return;
  try{const group=(await request(`/webrtc/groups/${encodeURIComponent(id)}`)).body;if((currentGroup?.group_id||groupId)!==id)return;const first=!currentGroup;adoptGroup(group);if(first){const ended=group.sessions.filter(s=>ENDED.includes(s.status)).length;setStatus(`Loaded group ${group.group_id} · ${plural(group.count,"peer")}.${ended?` ${ended} ended; Reconnect peers replaces them.`:""}`,ended?"warn":"ok")}else if(!silent)setStatus("Group refreshed.","ok")}
  catch(error){if(error.status===404&&(currentGroup?.group_id||groupId)===id){currentGroup=null;groupId=null;renderEndpoints();renderWall();syncControls();setStatus("This group no longer exists on the server. Create a new group.","warn")}else if(!silent)setStatus(error.message,"err")}
}
async function deleteGroup(silent=false) {
  if(!currentGroup)return setStatus("No group to delete.","warn");
  const id=currentGroup.group_id; let count=0;
  try{count=(await request(`/webrtc/groups/${encodeURIComponent(id)}`,{method:"DELETE"})).body.deleted_sessions.length}catch(error){if(error.status!==404)throw error}
  currentGroup=null; groupId=null; audioLeader=null; pendingRetry=null; sendErrors.clear(); clientStats.clear();
  if(initialMatch) history.replaceState(null,"","/webrtc/wall");
  renderEndpoints(); renderWall(); syncControls(); renderMetrics(); if(!silent)setStatus(`Deleted ${plural(count,"session")}.`,"ok");
}

async function streamBlob(blob,filename,sessionIds=null,turnId=null) {
  if(!currentGroup)throw new Error("Create a group first.");
  const id=currentGroup.group_id; const turn=turnId||`wall_${Date.now()}_${crypto.getRandomValues(new Uint32Array(1))[0].toString(36)}`;
  const form=new FormData(); form.append("audio_file",blob,filename); form.append("turn_id",turn); if(sessionIds)form.append("session_ids",sessionIds.join(","));
  const targets=sessionIds||sessions().map(s=>s.session_id);
  setStatus(`Submitting audio to ${plural(targets.length,"peer")}…`,"warn");
  let body;
  try{body=(await request(`/webrtc/groups/${encodeURIComponent(id)}/stream`,{method:"POST",body:form})).body}
  catch(error){
    // A lost response may follow server acceptance; the same turn ID never duplicates speech.
    if(!error.status||error.status===429||error.status>=500){pendingRetry={blob,filename,turnId:turn,sessionIds:targets};syncControls()}
    throw error;
  }
  for(const result of body.results) if(result.ok) sendErrors.delete(result.session_id); else sendErrors.set(result.session_id,`Send failed: ${result.detail||result.status}`);
  const failed=body.results.filter(result=>!result.ok).map(result=>result.session_id);
  pendingRetry=failed.length?{blob,filename,turnId:body.turn_id,sessionIds:failed}:null;
  const duplicates=body.results.filter(result=>result.status==="duplicate").length;
  setStatus(`Started ${body.started}; failed ${body.failed}.${duplicates?` ${duplicates} already had this turn (not repeated).`:""}${failed.length?" Retry failed peers reuses the same turn ID, so no peer hears it twice.":""}`,failed.length?"warn":"ok");
  await refreshGroup(true);
}
async function synthesize() {
  const text=$("ttsText").value.trim(); if(!text)throw new Error("Enter text for Kokoro.");
  setStatus("Synthesizing with local Kokoro… first use downloads and warms the model.","warn");
  const response=await send("/webrtc/tts/kokoro",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({text,voice:$("ttsVoice").value,speed:Number($("ttsSpeed").value)})});
  const blob=await response.blob();
  lastTts={ms:Number(response.headers.get("X-Kokoro-Synthesis-Ms")),audio:Number(response.headers.get("X-Kokoro-Audio-Seconds")),rtf:Number(response.headers.get("X-Kokoro-Real-Time-Factor")),cold:response.headers.get("X-Kokoro-Cold-Start")==="1"};
  return blob;
}
async function startKokoro(sessionIds=null) { if(!currentGroup)await createGroup(); const blob=await synthesize(); await refreshMetrics(); await streamBlob(blob,"kokoro_test.wav",sessionIds); }
async function startUpload() {
  const file=$("audioFile").files[0]; if(!file)return setStatus("Choose an audio file.","warn");
  if(file.size>20*1024*1024)throw new Error("Audio must be no larger than 20 MiB.");
  if(!currentGroup)await createGroup(); await streamBlob(file,file.name);
}
async function retryFailed() {
  if(!pendingRetry)return setStatus("Nothing to retry.","warn");
  const retry=pendingRetry; await streamBlob(retry.blob,retry.filename,retry.sessionIds,retry.turnId);
  if(!pendingRetry)setStatus("Retries accepted without duplicate turns.","ok");
}
async function interruptPeers(sessionIds=null) {
  if(!currentGroup)return setStatus("No group to interrupt.","warn");
  const body=(await request(`/webrtc/groups/${encodeURIComponent(currentGroup.group_id)}/interrupt`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(sessionIds?{session_ids:sessionIds}:{})})).body;
  for(const result of body.results) if(result.ok) sendErrors.delete(result.session_id); else sendErrors.set(result.session_id,`Interrupt failed: ${result.detail||result.status}`);
  if(pendingRetry&&!sessionIds) pendingRetry=null;
  if(pendingRetry&&sessionIds){const left=pendingRetry.sessionIds.filter(id=>!sessionIds.includes(id));pendingRetry=left.length?{...pendingRetry,sessionIds:left}:null}
  setStatus(body.failed?`${plural(body.failed,"interruption")} failed. See tile errors.`:"Speech interrupted. Connections remain live.",body.failed?"warn":"ok");
  await refreshGroup(true);
}
async function reconnect() {
  if(!currentGroup)return setStatus("No group to reconnect.","warn");
  const body=(await request(`/webrtc/groups/${encodeURIComponent(currentGroup.group_id)}/reconnect`,{method:"POST"})).body;
  for(const result of body.reconnect){if(result.previous_session_id){if(audioLeader===result.previous_session_id)audioLeader=result.session_id;sendErrors.delete(result.previous_session_id);clientStats.delete(result.previous_session_id)}if(!result.ok)sendErrors.set(result.session_id,`Reconnect failed: ${result.detail||result.status}`)}
  adoptGroup(body);
  for(const session of sessions()) postToFrame(session.session_id,{type:"webrtc-start",muted:session.session_id!==audioLeader});
  const count=status=>body.reconnect.filter(result=>result.status===status).length; const failed=body.reconnect.filter(result=>!result.ok).length;
  setStatus(`Reconnected: ${count("kept")} kept, ${count("replaced")} replaced with new calls${failed?`, ${failed} failed`:""}.`,failed?"warn":"ok");
}
function cycleDebug() { debugMode=debugModes[(debugModes.indexOf(debugMode)+1)%debugModes.length];writePref("webrtcWallDebugMode",debugMode);$("debugBtn").textContent=`Stats: ${debugMode}`;for(const session of sessions())postToFrame(session.session_id,{type:"webrtc-debug-mode",mode:debugMode}); }

window.addEventListener("message",event=>{
  if(event.origin!==location.origin)return; const data=event.data||{};
  if(data.type==="webrtc-token-request"){event.source?.postMessage({type:"webrtc-token",token:token()},location.origin);return}
  if(data.type!=="webrtc-client-stats")return;
  clientStats.set(data.sessionId,data.stats||{});
  const session=sessions().find(s=>s.session_id===data.sessionId); const card=document.querySelector(`.card[data-session-id="${CSS.escape(data.sessionId)}"]`); if(session&&card)updateCard(card,session);
});
$("createBtn").addEventListener("click",()=>run(createGroup)); $("reconnectBtn").addEventListener("click",()=>run(reconnect)); $("debugBtn").addEventListener("click",cycleDebug);
$("refreshBtn").addEventListener("click",()=>run(async()=>{await refreshServer();await refreshGroup(false);if(!currentGroup)setStatus(health?.ready?"Server ready.":"GPU server is not ready.",health?.ready?"ok":"warn")}));
$("interruptBtn").addEventListener("click",()=>run(()=>interruptPeers())); $("deleteBtn").addEventListener("click",()=>run(()=>deleteGroup()));
$("uploadBtn").addEventListener("click",()=>run(startUpload)); $("kokoroBtn").addEventListener("click",()=>run(()=>startKokoro())); $("retryBtn").addEventListener("click",()=>run(retryFailed));
// Not run(): blurring the token field fires "change" as the user clicks a button;
// taking the busy lock here would disable that button mid-click and drop it.
async function applyToken() {
  try{await refreshServer();const had=Boolean(currentGroup);await refreshGroup(true);if(had||!currentGroup)setStatus(currentGroup?`Token accepted · group ${currentGroup.group_id}.`:(health?.ready?"Token accepted · server ready.":"GPU server is not ready."),health?.ready?"ok":"warn")}
  catch(error){setStatus(error.message,"err")}
}
$("token").addEventListener("change",applyToken);
$("debugBtn").textContent=`Stats: ${debugMode}`; syncControls(); renderWall();
refreshServer().then(()=>refreshGroup(true)).catch(error=>setStatus(error.message,"err"));
setInterval(async()=>{if(polling)return;polling=true;try{if(currentGroup||groupId)await refreshGroup(true);await refreshMetrics()}finally{polling=false}},2000);
