(function () {
  'use strict';

  var MAX_MB = 25;
  var PALETTE = ['--sp0', '--sp1', '--sp2', '--sp3', '--sp4', '--sp5', '--sp6', '--sp7'];
  // Set window.BIRD_API_BASE (for example "https://my-api.example.com/") only if
  // the page is hosted on a different domain than the API.
  var API_URL = (window.BIRD_API_BASE || '') + 'api/predict';

  var $ = function (id) { return document.getElementById(id); };
  var el = {
    form: $('form'), drop: $('drop'), file: $('file'), picked: $('picked'),
    pickedName: $('picked-name'), go: $('go'), error: $('error'),
    result: $('result'), demo: $('demo-banner'), heading: $('file-heading'),
    player: $('player'), threshold: $('threshold'), thresholdOut: $('threshold-out'),
    stack: $('stack'), wave: $('wave'), spec: $('spec'), lanes: $('lanes'),
    ruler: $('ruler'), playhead: $('playhead'), summary: $('summary'),
    table: $('windows-table'), limits: $('limits')
  };

  var state = { file: null, data: null, threshold: 0.6, audioUrl: null, raf: null };

  // ---------- helpers ----------
  function fmtTime(s) {
    var m = Math.floor(s / 60);
    var sec = Math.floor(s % 60);
    return m + ':' + (sec < 10 ? '0' : '') + sec;
  }
  function fmtRange(a, b) { return fmtTime(a) + '–' + fmtTime(b); }
  function pct(v) { return Math.round(v * 100) + '%'; }
  function colorVar(species) {
    var i = state.data.classes.indexOf(species);
    return 'var(' + PALETTE[(i < 0 ? 0 : i) % PALETTE.length] + ')';
  }
  function label(species) { return (state.data.labels && state.data.labels[species]) || species; }
  function node(tag, className, text) {
    var n = document.createElement(tag);
    if (className) n.className = className;
    if (text !== undefined) n.textContent = text;
    return n;
  }
  function dot(species) {
    var d = node('span', 'dot');
    d.style.background = colorVar(species);
    return d;
  }
  function showError(msg) { el.error.textContent = msg; el.error.hidden = false; }
  function clearError() { el.error.hidden = true; el.error.textContent = ''; }

  // ---------- choosing a file ----------
  function setFile(f) {
    clearError();
    if (!f) return;
    if (f.size > MAX_MB * 1024 * 1024) {
      showError('That file is ' + (f.size / 1048576).toFixed(1) + ' MB. The limit is ' + MAX_MB + ' MB.');
      return;
    }
    state.file = f;
    el.pickedName.textContent = f.name + ' (' + (f.size / 1048576).toFixed(1) + ' MB)';
    el.picked.hidden = false;
  }

  el.file.addEventListener('change', function () { setFile(el.file.files[0]); });
  ['dragenter', 'dragover'].forEach(function (ev) {
    el.drop.addEventListener(ev, function (e) { e.preventDefault(); el.drop.classList.add('over'); });
  });
  ['dragleave', 'drop'].forEach(function (ev) {
    el.drop.addEventListener(ev, function (e) { e.preventDefault(); el.drop.classList.remove('over'); });
  });
  el.drop.addEventListener('drop', function (e) {
    if (e.dataTransfer && e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
  });

  // ---------- sending it to the backend ----------
  el.form.addEventListener('submit', function (e) {
    e.preventDefault();
    if (!state.file) return;
    clearError();
    el.go.disabled = true;
    el.go.textContent = 'Analyzing…';

    var body = new FormData();
    body.append('file', state.file);
    body.append('hop', '2.5');
    body.append('threshold', String(state.threshold));

    fetch(API_URL, { method: 'POST', body: body })
      .then(function (res) {
        return res.json().catch(function () { return null; }).then(function (data) {
          if (!res.ok) {
            throw new Error((data && data.error) || 'The server answered with status ' + res.status + '.');
          }
          return data;
        });
      })
      .then(function (data) { state.data = data; render(); })
      .catch(function (err) {
        showError(err instanceof TypeError
          ? 'Could not reach the server. Check your connection and try again.'
          : err.message);
      })
      .then(function () {
        el.go.disabled = false;
        el.go.textContent = 'Analyze recording';
      });
  });

  // ---------- drawing the results ----------
  function render() {
    var d = state.data;
    state.threshold = d.threshold;
    el.threshold.value = String(d.threshold);
    el.thresholdOut.textContent = pct(d.threshold);

    el.demo.hidden = !d.demo;
    el.heading.textContent = d.filename || 'Recording';

    if (state.audioUrl) URL.revokeObjectURL(state.audioUrl);
    state.audioUrl = URL.createObjectURL(state.file);
    el.player.src = state.audioUrl;

    el.spec.src = 'data:image/png;base64,' + d.spectrogram_png;
    buildLanes();
    buildRuler();
    el.result.hidden = false;
    drawWaveform();
    updateDetections();
    buildTable();
    buildLimits();

    el.stack.classList.add('reveal');
    setTimeout(function () { el.stack.classList.remove('reveal'); }, 1000);
    setPlayhead(0);
    el.playhead.hidden = false;
    el.result.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function buildLanes() {
    el.lanes.textContent = '';
    state.data.classes.concat(['__unsure']).forEach(function (sp) {
      var row = node('div', 'track');
      var lab = node('div', 'label');
      if (sp === '__unsure') {
        lab.textContent = 'Below threshold';
      } else {
        lab.appendChild(dot(sp));
        lab.appendChild(document.createTextNode(label(sp)));
      }
      var lane = node('div', 'lane');
      lane.dataset.species = sp;
      row.appendChild(lab);
      row.appendChild(lane);
      el.lanes.appendChild(row);
    });
  }

  function buildRuler() {
    var dur = state.data.duration;
    var steps = [1, 2, 5, 10, 15, 30, 60];
    var step = steps[steps.length - 1];
    for (var i = 0; i < steps.length; i++) {
      if (dur / steps[i] <= 12) { step = steps[i]; break; }
    }
    el.ruler.textContent = '';
    for (var t = 0; t <= dur + 1e-6; t += step) {
      var tick = node('div', 'tick');
      tick.style.left = (t / dur * 100) + '%';
      tick.appendChild(node('span', null, fmtTime(t)));
      el.ruler.appendChild(tick);
    }
  }

  function placeBand(lane, start, end, className, tip, species) {
    var dur = state.data.duration;
    var b = node('div', 'band' + (className ? ' ' + className : ''));
    b.style.left = (start / dur * 100) + '%';
    b.style.width = ((end - start) / dur * 100) + '%';
    if (species) b.style.background = colorVar(species);
    b.title = tip;
    lane.appendChild(b);
  }

  function updateDetections() {
    var d = state.data;
    var threshold = state.threshold;
    var segs = BirdSegments.mergeSegments(d.windows, threshold);

    var lanes = {};
    el.lanes.querySelectorAll('.lane').forEach(function (l) {
      l.textContent = '';
      lanes[l.dataset.species] = l;
    });

    segs.forEach(function (s) {
      placeBand(lanes[s.species], s.start, s.end, '',
        label(s.species) + ', ' + fmtRange(s.start, s.end) + ', peak ' + pct(s.peak_confidence), s.species);
    });
    d.windows.forEach(function (w) {
      if (w.confidence < threshold) {
        placeBand(lanes.__unsure, w.start, w.end, 'unsure',
          fmtRange(w.start, w.end) + ': best guess ' + label(w.species) + ' at only ' + pct(w.confidence));
      }
    });

    renderSummary(BirdSegments.summarize(segs));
    renderTableDim();
  }

  function renderSummary(rows) {
    el.summary.textContent = '';
    if (!rows.length) {
      el.summary.appendChild(node('p', 'none',
        'No species reached ' + pct(state.threshold) + ' confidence. Lower the threshold to see the model’s best guesses.'));
      return;
    }
    rows.forEach(function (r) {
      var row = node('div', 'summary-row');
      row.appendChild(dot(r.species));
      row.appendChild(node('strong', null, label(r.species)));
      row.appendChild(node('span', 'detail',
        r.seconds.toFixed(1) + ' s in ' + r.segments + (r.segments === 1 ? ' segment' : ' segments') +
        ', peak confidence ' + pct(r.peak_confidence)));
      el.summary.appendChild(row);
    });
  }

  function buildTable() {
    var t = el.table;
    t.textContent = '';
    var head = node('tr');
    ['Time', 'Best guess', 'Confidence'].forEach(function (h) { head.appendChild(node('th', null, h)); });
    t.appendChild(head);
    state.data.windows.forEach(function (w) {
      var tr = node('tr');
      tr.appendChild(node('td', null, fmtRange(w.start, w.end)));
      var sp = node('td');
      sp.appendChild(dot(w.species));
      sp.appendChild(document.createTextNode(' ' + label(w.species)));
      tr.appendChild(sp);
      var c = node('td');
      var bar = node('span', 'bar');
      bar.style.width = Math.round(w.confidence * 80) + 'px';
      c.appendChild(bar);
      c.appendChild(document.createTextNode(pct(w.confidence)));
      tr.appendChild(c);
      t.appendChild(tr);
    });
  }

  function renderTableDim() {
    var rows = el.table.querySelectorAll('tr');
    state.data.windows.forEach(function (w, i) {
      rows[i + 1].className = w.confidence < state.threshold ? 'dim' : '';
    });
  }

  function buildLimits() {
    var names = state.data.classes.map(label).join(', ');
    el.limits.textContent =
      'The model knows ' + state.data.classes.length + ' species (' + names + '). It has no “no bird” class, ' +
      'so every window gets one of them as its best guess. Windows under the threshold are marked as uncertain. ' +
      'If two species call at the same time, each window still gets a single best guess.';
  }

  function drawWaveform() {
    var canvas = el.wave;
    var peaks = state.data.waveform;
    var dpr = window.devicePixelRatio || 1;
    var w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w) return;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    var ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--slate') || '#566872';
    var barW = Math.max(1, w / peaks.length);
    for (var i = 0; i < peaks.length; i++) {
      var y1 = (1 - peaks[i][1]) * h / 2;
      var y2 = (1 - peaks[i][0]) * h / 2;
      ctx.fillRect(i * w / peaks.length, y1, barW, Math.max(1, y2 - y1));
    }
  }
  if (window.ResizeObserver) new ResizeObserver(function () { if (state.data) drawWaveform(); }).observe(el.wave);

  // ---------- threshold, playhead, seeking ----------
  el.threshold.addEventListener('input', function () {
    state.threshold = parseFloat(el.threshold.value);
    el.thresholdOut.textContent = pct(state.threshold);
    if (state.data) updateDetections();
  });

  function setPlayhead(seconds) {
    var dur = state.data ? state.data.duration : 0;
    var frac = dur ? Math.min(1, Math.max(0, seconds / dur)) : 0;
    el.stack.style.setProperty('--p', String(frac));
  }
  function loop() {
    setPlayhead(el.player.currentTime);
    state.raf = requestAnimationFrame(loop);
  }
  el.player.addEventListener('play', function () { if (!state.raf) loop(); });
  ['pause', 'ended'].forEach(function (ev) {
    el.player.addEventListener(ev, function () {
      cancelAnimationFrame(state.raf);
      state.raf = null;
      setPlayhead(el.player.currentTime);
    });
  });
  el.player.addEventListener('seeked', function () { setPlayhead(el.player.currentTime); });

  el.stack.addEventListener('click', function (e) {
    if (!state.data) return;
    var rect = el.stack.getBoundingClientRect();
    var gutter = parseFloat(getComputedStyle(el.stack).getPropertyValue('--gutter')) || 0;
    var plotLeft = rect.left + gutter;
    var plotWidth = rect.width - gutter - 14;
    var frac = (e.clientX - plotLeft) / plotWidth;
    if (frac < 0 || frac > 1) return;
    el.player.currentTime = frac * state.data.duration;
    setPlayhead(el.player.currentTime);
  });
})();
