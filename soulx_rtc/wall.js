'use strict';
const $ = id => document.getElementById(id);
const peers = new Map();
let epoch = 0, busy = false, stopping = false, health = null, kokoro = null, leader = null;
const fmt = (v, digits = 1) => Number.isFinite(v) ? v.toFixed(digits) : '—';
const json = body => ({method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
function status(message, error = false) {
  $('status').textContent = message;
  $('status').classList.toggle('error', error);
}
async function request(path, options = {}) {
  const response = await fetch(path, {...options, cache: 'no-store'});
  if (!response.ok) {
    const error = new Error(`${response.status}: ${(await response.text()).slice(0, 600)}`);
    error.status = response.status;
    throw error;
  }
  return response;
}
async function api(path, options, credential) {
  const response = await request(path, options, credential);
  return response.status === 204 ? null : response.json();
}
function active(p) { return peers.get(p.key) === p && !p.closed; }
function controls() {
  const connected = [...peers.values()].filter(p => p.pc?.connectionState === 'connected' && !p.closed);
  $('create').disabled = busy || stopping || peers.size > 0;
  $('speak').disabled = busy || stopping || kokoro?.available === false;
  $('sendFile').disabled = busy || stopping;
  $('retry').disabled = busy || stopping || ![...peers.values()].some(p => p.pending);
  $('interrupt').disabled = busy || stopping || !connected.length;
  $('stop').disabled = stopping || (!busy && !peers.size);
  for (const id of ['count', 'seed']) $(id).disabled = busy || stopping || peers.size > 0;
  for (const button of document.querySelectorAll('.avatar-option')) button.disabled = busy || stopping || peers.size > 0;
  for (const p of peers.values()) {
    const ready = p.pc?.connectionState === 'connected' && !p.closed;
    p.speak.disabled = busy || stopping || !ready || !!p.pending || kokoro?.available === false;
    p.interrupt.disabled = busy || stopping || !ready;
    p.stop.disabled = stopping;
  }
  $('connected').textContent = `${connected.length} / ${peers.size}`;
  $('empty').hidden = peers.size > 0;
}
async function action(fn) {
  if (busy || stopping) return;
  busy = true;
  controls();
  const run = epoch;
  try { await fn(run); }
  catch (error) { if (run === epoch) status(error.message, true); }
  finally { busy = false; controls(); }
}
function makeTile(index, seed, credential) {
  const card = document.createElement('article');
  card.className = 'tile';
  // This template is constant; IDs, errors and server data use textContent.
  card.innerHTML = `<div class="tile-head"><h2></h2><span class="badge">Preparing…</span></div>
    <video autoplay playsinline muted></video><div class="actions">
    <button data-action="listen">Listen</button><button data-action="speak">Speak</button>
    <button data-action="interrupt">Interrupt</button><button data-action="stop" class="danger">Stop</button></div>
    <div class="notice identity"></div><div class="status error" role="status"></div>
    <div class="stats">Waiting for media…</div><details><summary>Server metrics</summary><pre></pre></details>`;
  const p = {key: `${epoch}:${index}`, index, seed, credential, avatarId: $('avatar').value, card, id: null, pc: null,
    closed: false, pending: null, metrics: null, previous: new Map(), fps: null, turn: null};
  p.video = card.querySelector('video');
  p.video.muted = true;
  p.badge = card.querySelector('.badge');
  p.error = card.querySelector('.status');
  for (const name of ['listen', 'speak', 'interrupt', 'stop']) p[name] = card.querySelector(`[data-action="${name}"]`);
  card.querySelector('h2').textContent = `Peer ${index + 1}`;
  card.querySelector('.identity').textContent = `Seed ${seed}`;
  p.listen.onclick = () => setLeader(leader === p.key ? null : p.key);
  p.speak.onclick = () => action(run => speak(run, [p]));
  p.interrupt.onclick = () => action(() => interruptPeers([p]));
  p.stop.onclick = () => { void closePeer(p).then(() => { controls(); totals(); }).catch(e => status(e.message, true)); };
  peers.set(p.key, p);
  $('wall').append(card);
  controls();
  return p;
}
function setLeader(key) {
  leader = key;
  for (const p of peers.values()) {
    const audible = p.key === key;
    p.video.muted = !audible;
    p.card.classList.toggle('audible', audible);
    p.listen.textContent = audible ? 'Audio on' : 'Listen';
    if (audible) p.video.play().then(() => { p.error.textContent = ''; }).catch(() => {
      p.error.textContent = 'Playback was blocked. Click Listen again to enable audio.';
      setLeader(null);
    });
  }
}
function waitPeer(pc, property, event, desired, timeoutMs) {
  return new Promise((resolve, reject) => {
    const finish = error => { clearTimeout(timer); pc.removeEventListener(event, check); pc.removeEventListener('connectionstatechange', check); error ? reject(error) : resolve(); };
    const check = () => {
      if (['failed', 'closed'].includes(pc.connectionState)) finish(new Error(`Peer ${pc.connectionState}`));
      else if (pc[property] === desired) finish();
    };
    const timer = setTimeout(() => finish(new Error(`${property} timed out. Check the server ICE/TURN configuration.`)), timeoutMs);
    pc.addEventListener(event, check);
    pc.addEventListener('connectionstatechange', check);
    check();
  });
}
async function deleteCall(id, credential) {
  if (id) await api(`/calls/${encodeURIComponent(id)}`, {method: 'DELETE'}, credential);
}
async function closePeer(p, remove = true) {
  p.closed = true;
  p.pending = null;
  p.pc?.close();
  if (p.video.srcObject) p.video.srcObject.getTracks().forEach(track => track.stop());
  p.video.srcObject = null;
  if (leader === p.key) setLeader(null);
  try {
    await deleteCall(p.id, p.credential);
    p.id = null;
  } catch (error) {
    // Retain the tile/ID so Stop can retry server cleanup after a network failure.
    p.error.textContent = `Release failed: ${error.message}. Click Stop to retry.`;
    throw error;
  }
  if (remove) { peers.delete(p.key); p.card.remove(); }
}
async function connect(p, config, run) {
  try {
    const call = await api('/calls', json({seed: p.seed, avatar_id: p.avatarId}), p.credential);
    p.id = call.id;
    if (run !== epoch || !active(p)) { await closePeer(p); return; }
    p.metrics = call;
    p.card.querySelector('.identity').textContent = `${call.avatar_name} · Seed ${p.seed} · ${p.id}`;
    const pc = p.pc = new RTCPeerConnection(config);
    p.video.srcObject = new MediaStream();
    pc.ontrack = event => {
      if (!active(p)) return;
      p.video.srcObject.addTrack(event.track);
      p.video.play().catch(() => { p.error.textContent = 'Click Listen to start playback.'; });
    };
    pc.onconnectionstatechange = () => {
      if (!active(p)) return;
      p.badge.textContent = pc.connectionState;
      if (pc.connectionState === 'failed') {
        p.error.textContent = 'Connection failed. Check ICE/TURN and recreate the wall.';
        void closePeer(p, false).catch(() => {}).finally(controls);
      }
      controls();
    };
    pc.addTransceiver('video', {direction: 'recvonly'});
    pc.addTransceiver('audio', {direction: 'recvonly'});
    await pc.setLocalDescription(await pc.createOffer());
    await waitPeer(pc, 'iceGatheringState', 'icegatheringstatechange', 'complete', 20000);
    if (run !== epoch || !active(p)) return;
    const answer = await api(`/calls/${p.id}/offer`, json({type: pc.localDescription.type, sdp: pc.localDescription.sdp}), p.credential);
    if (run !== epoch || !active(p)) return;
    await pc.setRemoteDescription(answer);
    await waitPeer(pc, 'connectionState', 'connectionstatechange', 'connected', 25000);
  } catch (error) {
    p.badge.textContent = 'Failed';
    p.error.textContent = error.message;
    await closePeer(p, false).catch(() => {});
    throw error;
  }
}
async function refreshServer() {
  const results = await Promise.allSettled([api('/health'), api('/webrtc/tts/kokoro/status'), api('/avatars'), api('/config')]);
  if (results[2].status === 'fulfilled') {
    const previous = $('avatar').value;
    const avatars = results[2].value.avatars.slice(0,4);
    const selected = avatars.some(a => a.id === previous) ? previous : avatars[0]?.id;
    $('avatar').value = selected || 'default';
    $('avatars').replaceChildren(...avatars.map(a => {
      const button=document.createElement('button');
      button.type='button';
      button.className='avatar-option'+(a.id === selected ? ' selected' : '');
      button.setAttribute('role','radio');
      button.setAttribute('aria-checked',a.id === selected ? 'true' : 'false');
      const image=document.createElement('img');
      image.src=a.preview_url;
      image.alt='';
      const name=document.createElement('span');
      name.textContent=a.name;
      button.append(image,name);
      button.onclick=() => {
        $('avatar').value=a.id;
        for (const option of document.querySelectorAll('.avatar-option')) {
          const active=option === button;
          option.classList.toggle('selected',active);
          option.setAttribute('aria-checked',active ? 'true' : 'false');
        }
      };
      return button;
    }));
  }
  if (results[0].status === 'fulfilled') {
    health = results[0].value;
    $('count').max = health.max_sessions;
    $('profile').textContent = `${health.width} × ${health.height} · ${health.steps} steps · ${health.fps} fps · batch ${health.batch} · memory: ${health.memory_mode} · up to ${health.max_active_calls} rendering calls / ${health.max_sessions} connected peers · idle: ${health.idle_policy} · avatar: ${health.idle_asset?.file || 'included sample'} · ${health.ready ? 'GPU ready' : 'GPU not ready'}`;
  } else { health = null; $('profile').textContent = results[0].reason.message; }
  if (results[1].status === 'fulfilled') {
    kokoro = results[1].value;
    const previous = $('voice').value;
    $('voice').replaceChildren(...kokoro.voices.map(voice => new Option(voice, voice)));
    if (kokoro.voices.includes(previous)) $('voice').value = previous;
    $('ttsStatus').textContent = kokoro.available ? 'Local CPU · first use downloads and warms Kokoro.' : 'Install requirements-tts.txt in the server environment to enable Kokoro.';
  } else { kokoro = null; $('ttsStatus').textContent = results[1].reason.message; }
  controls();
}
async function createWall(run) {
  if (peers.size) return;
  await refreshServer();
  if (run !== epoch) return;
  if (!health?.ready) throw new Error('GPU server is not ready.');
  const count = Number($('count').value), seed = Number($('seed').value);
  if (!Number.isInteger(count) || count < 1 || count > health.max_sessions) throw new Error(`Choose 1–${health.max_sessions} peers.`);
  if (!Number.isSafeInteger(seed) || seed < 0 || seed > 2147483647) throw new Error('First seed must be an integer from 0 to 2147483647.');
  const credential = '', config = await api('/config');
  if (run !== epoch) return;
  status(`Connecting ${count} independent calls…`);
  const tiles = Array.from({length: count}, (_, i) => makeTile(i, seed + i, credential));
  const results = await Promise.allSettled(tiles.map(p => connect(p, config, run)));
  if (run !== epoch) return;
  const failed = results.filter(r => r.status === 'rejected').length;
  status(`${count - failed}/${count} peers connected. Select Listen on one tile to enable audio.${failed ? ' Failed tiles show the cause; Stop releases them.' : ''}`, failed > 0);
  controls();
}
async function deliver(p) {
  if (!active(p) || !p.pending) return;
  const pending = p.pending;
  try {
    await api(`/calls/${p.id}/turns`, {method: 'POST', headers: {'X-Turn-ID': pending.id, 'Content-Type': pending.blob.type || 'application/octet-stream'}, body: pending.blob}, p.credential);
    if (active(p) && p.pending === pending) {
      p.turn = pending.id;
      p.pending = null;
      p.error.textContent = '';
    }
  } catch (error) {
    if (active(p)) p.error.textContent = `${error.message} — Retry failed sends reuses this turn ID.`;
    throw error;
  }
}
async function broadcast(blob, targets, run) {
  if (run !== epoch) return;
  const live = targets.filter(p => active(p) && p.pc?.connectionState === 'connected');
  if (!live.length) throw new Error('No connected peers. Create the wall first.');
  if (live.some(p => p.pending)) throw new Error('Retry the failed sends or interrupt those peers before sending new speech.');
  const id = `wall-${Date.now()}-${crypto.getRandomValues(new Uint32Array(2)).join('-')}`;
  for (const p of live) p.pending = {id, blob};
  const results = await Promise.allSettled(live.map(deliver));
  if (run !== epoch) return;
  const failed = results.filter(r => r.status === 'rejected').length;
  status(`Speech accepted by ${live.length - failed}/${live.length} peers.${failed ? ' Use Retry failed sends; successful peers will not receive duplicates.' : ' Send another turn at any time, or interrupt playback.'}`, failed > 0);
}
async function speak(run, targets = null) {
  if (!targets && !peers.size) await createWall(run);
  if (run !== epoch) return;
  targets = targets || [...peers.values()];
  if (!targets.some(p => active(p) && p.pc?.connectionState === 'connected')) throw new Error('No connected peers. Recreate the wall.');
  if (targets.some(p => p.pending)) throw new Error('Retry failed sends or interrupt before synthesizing another turn.');
  if (!$('text').value.trim()) throw new Error('Enter text for Kokoro.');
  status('Synthesizing Kokoro speech on CPU… first use may take a minute to download the model.');
  const response = await request('/webrtc/tts/kokoro', json({text: $('text').value, voice: $('voice').value, speed: Number($('speed').value)}));
  const blob = await response.blob();
  if (run !== epoch) return;
  $('ttsMetric').textContent = `${fmt(Number(response.headers.get('X-Kokoro-Synthesis-Ms')) / 1000, 2)}s / ${fmt(Number(response.headers.get('X-Kokoro-Audio-Seconds')), 2)}s`;
  $('ttsStatus').textContent = `CPU · RTF ${response.headers.get('X-Kokoro-Real-Time-Factor')} · ${response.headers.get('X-Kokoro-Cold-Start') === '1' ? 'cold start' : 'warm'}`;
  await broadcast(blob, targets, run);
}
async function interruptPeers(targets) {
  const results = await Promise.allSettled(targets.filter(p => active(p) && p.id).map(async p => {
    try {
      await api(`/calls/${p.id}/interrupt`, json({}), p.credential);
      p.pending = null; p.turn = null; p.error.textContent = '';
    } catch (error) { p.error.textContent = error.message; throw error; }
  }));
  const failed = results.filter(r => r.status === 'rejected').length;
  status(failed ? `${failed} interruption(s) failed. See tile errors.` : 'Speech interrupted. Connections remain live.', !!failed);
}
async function stopAll() {
  if (stopping) return;
  stopping = true;
  epoch++;
  controls();
  const results = await Promise.allSettled([...peers.values()].map(p => closePeer(p)));
  const failed = results.filter(r => r.status === 'rejected').length;
  stopping = false;
  status(failed ? 'Some calls could not be released. Stop again to retry.' : 'Wall stopped. Calls released; any pending preparation will be released when it returns.', !!failed);
  controls(); totals();
}
function totals() {
  const all = [...peers.values()];
  $('speaking').textContent = `${all.filter(p => p.metrics?.turns?.some(t => ['speaking', 'draining'].includes(t.status) && !t.synthetic_idle)).length} / ${all.filter(p => p.metrics?.queued_turns > 0).length}`;
  const values = all.filter(p => !p.closed && Number.isFinite(p.fps)).map(p => p.fps);
  $('fps').textContent = values.length ? fmt(values.reduce((a, b) => a + b, 0) / values.length) : '—';
}
async function pollPeer(p) {
  if (!active(p) || !p.id || !p.pc) return;
  const results = await Promise.allSettled([api(`/calls/${p.id}`, {}, p.credential), p.pc.getStats()]);
  if (!active(p)) return;
  if (results[0].status === 'fulfilled') p.metrics = results[0].value;
  else {
    p.error.textContent = `Stats: ${results[0].reason.message}`;
    if (results[0].reason.status === 404) { await closePeer(p, false).catch(() => {}); p.badge.textContent = 'Closed'; return; }
  }
  const m = p.metrics, lines = [];
  if (results[1].status === 'fulfilled') {
    const stats = results[1].value;
    for (const r of stats.values()) {
      if (r.type !== 'inbound-rtp' || r.isRemote) continue;
      const kind = r.kind || r.mediaType, previous = p.previous.get(r.id);
      const seconds = previous ? (r.timestamp - previous.timestamp) / 1000 : 0;
      const fps = previous && seconds > 0 ? (r.framesDecoded - previous.framesDecoded) / seconds : r.framesPerSecond;
      if (kind === 'video') p.fps = fps;
      const emitted = previous ? r.jitterBufferEmittedCount - previous.jitterBufferEmittedCount : 0;
      const buffer = emitted > 0 ? (r.jitterBufferDelay - previous.jitterBufferDelay) * 1000 / emitted : null;
      const codec = stats.get(r.codecId)?.mimeType || kind;
      lines.push(`${codec} · ${kind === 'video' ? `${fmt(fps)} fps · drops ${r.framesDropped ?? '—'} · ` : ''}lost ${r.packetsLost ?? '—'}`);
      lines.push(`  jitter ${fmt(r.jitter * 1000)} ms · buffer ${fmt(buffer)} ms`);
      p.previous.set(r.id, r);
    }
    const transport = [...stats.values()].find(r => r.type === 'transport' && r.selectedCandidatePairId);
    const pair = stats.get(transport?.selectedCandidatePairId);
    if (pair) lines.push(`RTT ${fmt(pair.currentRoundTripTime * 1000)} ms · ${stats.get(pair.localCandidateId)?.candidateType || '?'} → ${stats.get(pair.remoteCandidateId)?.candidateType || '?'}`);
  }
  if (m) {
    const turn = m.turns?.find(t => t.id === p.turn);
    lines.push(`Neural frames ${m.generated_frames} · idle ${m.idle_frames} · held ${m.held_frames}`);
    lines.push(`Speech audio holds ${fmt(m.audio_hold_samples / 48000, 2)} s · queued turns ${m.queued_turns}`);
    lines.push(`Last turn ${turn?.status || '—'} · first sender media ${fmt(turn?.first_media_s, 2)} s`);
    p.badge.textContent = `${p.pc.connectionState} · ${m.active_turn ? (m.turns.find(t => t.id === m.active_turn)?.status || 'neural idle') : 'idle'}`;
    p.card.querySelector('pre').textContent = JSON.stringify(m, null, 2);
    if (m.error) p.error.textContent = `Server: ${m.error}`;
  }
  p.card.querySelector('.stats').textContent = lines.join('\n');
}
async function poll() {
  try { await Promise.allSettled([...peers.values()].map(pollPeer)); totals(); controls(); }
  finally { setTimeout(poll, 1000); }
}
$('create').onclick = () => action(createWall);
$('refresh').onclick = () => action(refreshServer);
$('speak').onclick = () => action(run => speak(run));
$('sendFile').onclick = () => action(async run => {
  const file = $('audio').files[0];
  if (!file) throw new Error('Choose an audio file first.');
  if (file.size > 20 * 1024 * 1024) throw new Error('Audio must be no larger than 20 MiB.');
  if (!peers.size) await createWall(run);
  await broadcast(file, [...peers.values()], run);
});
$('retry').onclick = () => action(async run => {
  const results = await Promise.allSettled([...peers.values()].filter(p => p.pending).map(deliver));
  if (run === epoch) status(results.some(r => r.status === 'rejected') ? 'Some sends still failed. See tile errors.' : 'Retries accepted without duplicate turns.', results.some(r => r.status === 'rejected'));
});
$('interrupt').onclick = () => action(() => interruptPeers([...peers.values()]));
$('stop').onclick = stopAll;
$('mute').onclick = () => setLeader(null);
window.addEventListener('pagehide', () => {
  epoch++;
  for (const p of peers.values()) {
    p.closed = true; p.pc?.close();
    if (p.id) void request(`/calls/${p.id}`, {method: 'DELETE', keepalive: true}, p.credential).catch(() => {});
  }
});
void refreshServer();
void poll();
