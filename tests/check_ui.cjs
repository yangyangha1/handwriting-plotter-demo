const fs = require('fs');
const vm = require('vm');
const assert = require('assert');
const elements = new Map();
const boxes = [0, 1, 2, 3, 4].map(index => ({dataset: {index: String(index)}, checked: true}));
function element(id) {
  if (!elements.has(id)) elements.set(id, {
    style: {}, dataset: {}, hidden: false, attributes: {}, events: {},
    classList: {toggle() {}, add() {}, remove() {}},
    setAttribute(k, v) { this.attributes[k] = v; },
    removeAttribute(k) { delete this.attributes[k]; },
    addEventListener(k, fn) { this.events[k] = fn; },
    focus() {}, parentElement: {setAttribute() {}},
  });
  return elements.get(id);
}
const context = {
  document: {getElementById: element, querySelector: () => null,
    querySelectorAll: selector => selector.includes('#review-body') ? boxes : []},
  window: {setTimeout() {}, clearTimeout() {}}, assert, boxes,
};
const html = fs.readFileSync('src/hwplotter/web/index.html', 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1]
  .replace('updateEngineHint(); setWorkflowProgress(0); setUnlocked(1); loadEnvironment();', '');
vm.runInNewContext(script + `
state.items = ['中', '，', '。', 'A', '!'].map((char, index) => ({char, index}));
setPunctuation(false);
assert.deepStrictEqual(Array.from(boxes, b => b.checked), [true, false, false, true, false]);
setPunctuation(true);
assert(boxes.every(b => b.checked));
startTraining();
setTrainingProgress('恢复笔画', 3, 10);
assert.strictEqual($('process-progress').hidden, true);
assert.strictEqual($('metric-completion').textContent, '3/10');
assert.strictEqual($('completion-bar').style.width, '30%');
assert.strictEqual($('completion-progress').attributes['aria-valuenow'], '30');
setTrainingProgress('模型训练');
assert.strictEqual($('completion-progress').attributes['aria-valuenow'], undefined);
renderTraining({coverage: {overall_completion: .42}, readiness: {ready: false}});
assert.strictEqual($('metric-completion').textContent, '已完成');
assert.strictEqual($('completion-bar').style.width, '100%');
assert.strictEqual($('metric-coverage').textContent, '42%');
assert.strictEqual($('coverage-bar').style.width, '42%');
assert.strictEqual(trainingBusy, false);
`, context);
console.log('UI interaction checks passed: punctuation, training placement, stage count, coverage result');
