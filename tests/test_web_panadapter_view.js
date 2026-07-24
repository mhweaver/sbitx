"use strict";

const assert = require("assert");
const PanadapterView = require("../web/panadapter-view.js");

const view = PanadapterView.create(25000, 64);
assert.strictEqual(PanadapterView.spanHz(view), 25000);
assert.strictEqual(PanadapterView.tunedX(view, 600), 300);
assert.strictEqual(PanadapterView.sourceBin(view, 0, 600, 2048), 2047);
assert.strictEqual(PanadapterView.sourceBin(view, 599, 600, 2048), 0);

PanadapterView.zoomAt(view, 2, 0.75);
assert.strictEqual(PanadapterView.spanHz(view), 12500);
assert.strictEqual(PanadapterView.tunedX(view, 600), 150);
assert.ok(Math.abs(PanadapterView.sourceBin(view, 450, 600, 2048) - 512) <= 3);

PanadapterView.pan(view, 10);
assert.strictEqual(view.center, 0.25);
PanadapterView.pan(view, -10);
assert.strictEqual(view.center, -0.25);

PanadapterView.reset(view);
assert.deepStrictEqual([view.zoom, view.center], [1, 0]);

console.log("web panadapter view tests passed");
