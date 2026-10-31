// Merges per-window predictions into detection segments.
// Mirrors merge_segments() and summarize() in inference.py; both are checked
// against tests/fixture_windows.json. The page uses this to re-draw detections
// instantly when the confidence threshold slider moves, without another upload.
(function (root) {
  function round(v, digits) {
    const f = Math.pow(10, digits);
    return Math.round(v * f) / f;
  }

  function mergeSegments(windows, threshold) {
    const segs = [];
    let cur = null;
    for (const w of windows) {
      if (w.confidence < threshold) { cur = null; continue; }
      if (cur && cur.species === w.species && w.start <= cur.end + 0.011) {
        cur.end = Math.max(cur.end, w.end);
        cur.confs.push(w.confidence);
      } else {
        cur = { species: w.species, start: w.start, end: w.end, confs: [w.confidence] };
        segs.push(cur);
      }
    }
    return segs.map((s) => ({
      species: s.species,
      start: round(s.start, 2),
      end: round(s.end, 2),
      mean_confidence: round(s.confs.reduce((a, b) => a + b, 0) / s.confs.length, 4),
      peak_confidence: round(Math.max.apply(null, s.confs), 4),
    }));
  }

  function unionSeconds(intervals) {
    const sorted = intervals.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    let total = 0, curStart = null, curEnd = null;
    for (const [a, b] of sorted) {
      if (curEnd === null || a > curEnd) {
        if (curEnd !== null) total += curEnd - curStart;
        curStart = a; curEnd = b;
      } else {
        curEnd = Math.max(curEnd, b);
      }
    }
    if (curEnd !== null) total += curEnd - curStart;
    return total;
  }

  function summarize(segments) {
    const by = new Map();
    for (const s of segments) {
      if (!by.has(s.species)) by.set(s.species, []);
      by.get(s.species).push(s);
    }
    const rows = [];
    for (const [species, segs] of by) {
      rows.push({
        species,
        segments: segs.length,
        seconds: round(unionSeconds(segs.map((s) => [s.start, s.end])), 2),
        peak_confidence: Math.max.apply(null, segs.map((s) => s.peak_confidence)),
      });
    }
    return rows.sort((a, b) => b.seconds - a.seconds);
  }

  const api = { mergeSegments, summarize };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.BirdSegments = api;
})(typeof self !== 'undefined' ? self : this);
