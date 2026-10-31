// Run with: node --test tests/
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { mergeSegments, summarize } = require('../static/segments.js');

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
