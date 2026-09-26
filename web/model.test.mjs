import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import {
  decodeData, pairValues, langIndex, commonTaskAverages, fitFixedEffects,
  naiveMean, ratioByLang, taskDetail,
} from "./model.js";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const data = JSON.parse(readFileSync(path.join(root, "build/data.json"), "utf8"));
const li = new Map(data.langs.map((l, i) => [l, i]));

// 1. Common-task averages vs an independent Python computation (o200k, first variant).
const expected = {
  Clojure: 111, Julia: 110, Ruby: 119, Perl: 125, Python: 133, Haskell: 133, "F#": 140, Lua: 142,
  Scala: 148, OCaml: 158, PHP: 165, JavaScript: 185, Java: 191, Rust: 202, Go: 208, "C#": 215,
  "C++": 269, C: 294,
};
const setLangs = data.defaultSet;
const idxs = setLangs.map((l) => { assert.ok(li.has(l), `missing ${l}`); return li.get(l); });
const common = commonTaskAverages(data, idxs, ["openai-o200k"], "first");
assert.equal(common.tasks.length, 243);
setLangs.forEach((l, k) => assert.equal(Math.round(common.avg["openai-o200k"][k]), expected[l], l));
console.log(`common-task averages: ${common.tasks.length} tasks, all ${setLangs.length} match`);

// 2. Fixed effects vs independent sparse solve.
const tok = "openai-o200k";
const values = pairValues(data, tok, "first");
let t0 = performance.now();
const fit = fitFixedEffects(data, values);
const ms = performance.now() - t0;
console.log(`fitFixedEffects: ${fit.iterations} iterations, ${ms.toFixed(0)} ms, rmse ${fit.rmse.toFixed(4)}`);
assert.ok(ms < 300, `fit took ${ms} ms`);
const ref = JSON.parse(execFileSync(path.join(root, ".venv/bin/python"),
  [path.join(root, "scripts/check_fe.py"), tok, "first"], { maxBuffer: 1 << 26 }).toString());
let worst = 0, compared = 0;
for (let l = 0; l < data.langs.length; l++) {
  if (fit.n[l] < 5) continue;
  const rel = Math.abs(fit.typical[l] / ref[l] - 1);
  worst = Math.max(worst, rel);
  compared++;
  assert.ok(rel < 0.005, `${data.langs[l]}: js ${fit.typical[l]} vs py ${ref[l]}`);
}
console.log(`fixed effects match numpy lstsq for ${compared} languages (n>=5), worst rel diff ${(worst * 100).toFixed(4)}%`);

const ranked = [...data.langs.keys()].filter((l) => fit.n[l] >= 100).sort((x, y) => fit.typical[x] - fit.typical[y]);
const fmt = (l) => `${data.langs[l]} ${fit.typical[l].toFixed(1)} (n=${fit.n[l]}, se=${(fit.se[l] * 100).toFixed(1)}%)`;
console.log(`\n${tok}/first typical tokens, n>=100 (${ranked.length} languages)`);
console.log("most efficient:\n  " + ranked.slice(0, 15).map(fmt).join("\n  "));
console.log("least efficient:\n  " + ranked.slice(-15).map(fmt).join("\n  "));

// Remaining exports: shape checks.
const ix = langIndex(data);
assert.equal(ix.pairsByLang.length, data.langs.length);
assert.equal(ix.nTasks.reduce((s, x) => s + x, 0), data.pairs.task.length);
const naive = naiveMean(data, values);
const cpt = ratioByLang(data, values, pairValues(data, "chars", "first"));
const py = li.get("Python");
console.log(`\nPython: naive mean ${naive[py].toFixed(1)}, typical ${fit.typical[py].toFixed(1)}, chars/token ${cpt[py].toFixed(2)}`);
const detail = taskDetail(data, py, values);
assert.equal(detail.length, fit.n[py]);
assert.ok(detail[0].ratio <= detail.at(-1).ratio);
const meanVals = pairValues(data, tok, "mean");
assert.ok(meanVals.every((v) => v > 0));

// 3. decodeData round trip of the embedded payload.
const html = readFileSync(path.join(root, "dist/index.html"), "utf8");
const open = '<script id="data" type="application/octet-stream">';
const start = html.indexOf(open) + open.length;
const b64 = html.slice(start, html.indexOf("</script>", start));
t0 = performance.now();
const decoded = await decodeData(b64);
console.log(`decodeData: ${(performance.now() - t0).toFixed(0)} ms`);
assert.ok(JSON.stringify(decoded) === JSON.stringify(data), "decoded payload differs from build/data.json");
console.log("decodeData round trip ok\nall tests passed");
