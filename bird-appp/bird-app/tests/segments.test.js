// Run with: node --test tests/segments.test.js
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { mergeSegments, summarize, detectionsAt, segmentsFromProbs } = require('../static/segments.js');

const fx = JSON.parse(fs.readFileSync(path.join(__dirname, 'fixture_windows.json'), 'utf8'));

test('mergeSegments matches the hand-worked fixture', () => {
  assert.deepEqual(mergeSegments(fx.windows, fx.threshold), fx.segments);
});

test('summarize matches the hand-worked fixture', () => {
  assert.deepEqual(summarize(fx.segments), fx.summary);
});

test('nothing above the threshold gives empty results', () => {
  assert.deepEqual(mergeSegments(fx.windows, 0.99), []);
  assert.deepEqual(summarize([]), []);
});

// ---- multi-label: several birds at once, and "no bird" ----
const ml = JSON.parse(fs.readFileSync(path.join(__dirname, 'fixture_multilabel.json'), 'utf8'));

test('detectionsAt lists every species over the threshold, or none', () => {
  const got = ml.windows.map((w) => detectionsAt(w, ml.classes, ml.threshold));
  assert.deepEqual(got, ml.detections);
});

test('detectionsAt has no best-guess fallback', () => {
  assert.deepEqual(detectionsAt(ml.windows[4], ml.classes, ml.threshold), []);
});

test('segmentsFromProbs matches the multi-label fixture and allows overlap', () => {
  const got = segmentsFromProbs(ml.windows, ml.classes, ml.threshold);
  assert.deepEqual(got, ml.segments);
  assert.ok(got[1].start < got[0].end, 'B starts while A is still going');
});

test('summarize matches the multi-label fixture', () => {
  assert.deepEqual(summarize(ml.segments), ml.summary);
});

test('raising the threshold only removes species', () => {
  assert.deepEqual(segmentsFromProbs(ml.windows, ml.classes, 0.78).map((s) => s.species), ['A']);
  assert.deepEqual(segmentsFromProbs(ml.windows, ml.classes, 0.99), []);
});
