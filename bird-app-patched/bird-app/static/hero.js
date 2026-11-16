/*
  Decorative hero only. Draws a slowly drifting "dawn chorus": short frequency
  sweeps, like bird calls look on a spectrogram. Not connected to any data.
*/
(function () {
  'use strict';

  var canvas = document.getElementById('hero-canvas');
  if (!canvas || !canvas.getContext) return;
  var ctx = canvas.getContext('2d');
  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // dark violet -> rose -> orange -> pale yellow, the same family as the real spectrogram
  var RAMP = [[74, 28, 120], [158, 40, 120], [226, 83, 74], [255, 160, 70], [255, 232, 160]];

  var W = 0, H = 0, dpr = 1, calls = [], last = 0, raf = null, visible = true;

  function rnd(a, b) { return a + Math.random() * (b - a); }

  function ramp(t) {
    t = Math.min(0.999, Math.max(0, t)) * (RAMP.length - 1);
    var i = Math.floor(t), f = t - i, a = RAMP[i], b = RAMP[i + 1];
    return [Math.round(a[0] + (b[0] - a[0]) * f), Math.round(a[1] + (b[1] - a[1]) * f), Math.round(a[2] + (b[2] - a[2]) * f)];
  }

  function seed() {
    calls = [];
    var n = Math.max(14, Math.round(W / 34));
    for (var i = 0; i < n; i++) {
      var kind = Math.random();
      calls.push({
        x: rnd(0, W * 1.15),
        y: rnd(0.12, 0.72) * H,
        w: kind < 0.5 ? rnd(26, 60) : rnd(70, 150),   // short chips and longer whistles
        rise: rnd(-0.22, 0.22) * H,                    // net sweep up or down
        bend: rnd(-0.12, 0.12) * H,                    // curvature
        speed: rnd(7, 16),
        phase: rnd(0, 6.28),
        harmonics: Math.random() < 0.6
      });
    }
  }

  function resize() {
    var r = canvas.getBoundingClientRect();
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    W = Math.max(1, r.width); H = Math.max(1, r.height);
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    seed();
    draw(0);
  }

  function path(c, x, yOff, scale) {
    ctx.beginPath();
    for (var s = 0; s <= 24; s++) {
      var u = s / 24;
      var px = x + u * c.w;
      var py = c.y * scale + yOff + c.rise * (u - 0.5) + c.bend * (4 * (u - 0.5) * (u - 0.5) - 1);
      if (s === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
  }

  function draw(t) {
    ctx.clearRect(0, 0, W, H);

    // faint frequency lines
    ctx.strokeStyle = 'rgba(255,255,255,0.045)';
    ctx.lineWidth = 1;
    for (var gy = 36; gy < H; gy += 36) {
      ctx.beginPath(); ctx.moveTo(0, gy + 0.5); ctx.lineTo(W, gy + 0.5); ctx.stroke();
    }

    ctx.lineCap = 'round';
    for (var i = 0; i < calls.length; i++) {
      var c = calls[i];
      var x = c.x - c.speed * t;
      var span = W + 200;
      x = ((x % span) + span) % span - 100;           // wrap around, off-screen on both sides
      var pulse = 0.55 + 0.45 * Math.sin(t * 0.8 + c.phase);
      var heat = 1 - c.y / H;                          // higher on screen = hotter colour
      var rgb = ramp(heat * 0.85 + 0.1);
      var col = rgb[0] + ',' + rgb[1] + ',' + rgb[2];

      ctx.strokeStyle = 'rgba(' + col + ',' + (0.16 * pulse).toFixed(3) + ')';
      ctx.lineWidth = 9;
      path(c, x, 0, 1); ctx.stroke();
      ctx.strokeStyle = 'rgba(' + col + ',' + (0.85 * pulse).toFixed(3) + ')';
      ctx.lineWidth = 2;
      path(c, x, 0, 1); ctx.stroke();

      if (c.harmonics) {
        ctx.strokeStyle = 'rgba(' + col + ',' + (0.3 * pulse).toFixed(3) + ')';
        ctx.lineWidth = 1.5;
        path(c, x, -H * 0.16, 1); ctx.stroke();
      }
    }
  }

  function frame(now) {
    raf = null;
    if (!visible) return;
    var t = now / 1000;
    if (now - last > 30) { draw(t); last = now; }
    raf = requestAnimationFrame(frame);
  }

  function start() { if (!reduce && !raf && visible && !document.hidden) raf = requestAnimationFrame(frame); }
  function stop() { if (raf) { cancelAnimationFrame(raf); raf = null; } }

  resize();
  if (window.ResizeObserver) new ResizeObserver(resize).observe(canvas);
  else window.addEventListener('resize', resize);

  if (!reduce) {
    document.addEventListener('visibilitychange', function () { document.hidden ? stop() : start(); });
    if (window.IntersectionObserver) {
      new IntersectionObserver(function (entries) {
        visible = entries[0].isIntersecting;
        visible ? start() : stop();
      }).observe(canvas);
    }
    start();
  }
})();