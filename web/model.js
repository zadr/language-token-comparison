export async function decodeData(b64) {
  const bin = atob(b64.trim());
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
  return JSON.parse(await new Response(stream).text());
}

export function pairValues(data, tokId, agg) {
  const p = data.pairs;
  const src = tokId === "chars"
    ? { first: p.chars_first, sum: p.chars_sum }
    : p.tok[tokId];
  if (!src) throw new Error(`unknown tokenizer ${tokId}`);
  const n = p.task.length;
  const out = new Float64Array(n);
  if (agg === "first") {
    for (let i = 0; i < n; i++) out[i] = src.first[i];
  } else if (agg === "mean") {
    for (let i = 0; i < n; i++) out[i] = src.sum[i] / p.nvar[i];
  } else {
    throw new Error(`unknown aggregation ${agg}`);
  }
  return out;
}

const INDEX = Symbol("langIndex");

export function langIndex(data) {
  if (data[INDEX]) return data[INDEX];
  const { task, lang } = data.pairs;
  const L = data.langs.length;
  const counts = new Int32Array(L);
  for (let i = 0; i < lang.length; i++) counts[lang[i]]++;
  const pairsByLang = Array.from(counts, (c) => new Int32Array(c));
  const fill = new Int32Array(L);
  const tasksByLang = Array.from({ length: L }, () => new Set());
  for (let i = 0; i < lang.length; i++) {
    const l = lang[i];
    pairsByLang[l][fill[l]++] = i;
    tasksByLang[l].add(task[i]);
  }
  const nTasks = Int32Array.from(tasksByLang, (s) => s.size);
  return (data[INDEX] = { pairsByLang, tasksByLang, nTasks });
}

export function commonTaskAverages(data, langIdxs, tokIds, agg) {
  const { tasksByLang, pairsByLang } = langIndex(data);
  let tasks = [];
  if (langIdxs.length) {
    const sets = langIdxs.map((l) => tasksByLang[l]).sort((a, b) => a.size - b.size);
    tasks = [...sets[0]].filter((t) => sets.every((s) => s.has(t))).sort((a, b) => a - b);
  }
  const common = new Set(tasks);
  const avg = {};
  for (const tokId of tokIds) {
    const values = pairValues(data, tokId, agg);
    avg[tokId] = Float64Array.from(langIdxs, (l) => {
      let s = 0;
      for (const i of pairsByLang[l]) if (common.has(data.pairs.task[i])) s += values[i];
      return tasks.length ? s / tasks.length : NaN;
    });
  }
  return { tasks, avg };
}

export function fitFixedEffects(data, values) {
  const { task, lang } = data.pairs;
  const N = task.length;
  const T = data.tasks.length;
  const L = data.langs.length;
  const y = new Float64Array(N);
  for (let i = 0; i < N; i++) y[i] = Math.log(values[i]);

  const nT = new Int32Array(T);
  const n = new Int32Array(L);
  for (let i = 0; i < N; i++) { nT[task[i]]++; n[lang[i]]++; }

  const a = new Float64Array(T);
  const b = new Float64Array(L);
  const acc = new Float64Array(Math.max(T, L));
  let iterations = 0;
  for (; iterations < 500; iterations++) {
    acc.fill(0, 0, T);
    for (let i = 0; i < N; i++) acc[task[i]] += y[i] - b[lang[i]];
    for (let t = 0; t < T; t++) a[t] = nT[t] ? acc[t] / nT[t] : 0;
    acc.fill(0, 0, L);
    for (let i = 0; i < N; i++) acc[lang[i]] += y[i] - a[task[i]];
    let delta = 0;
    for (let l = 0; l < L; l++) {
      const next = n[l] ? acc[l] / n[l] : 0;
      delta = Math.max(delta, Math.abs(next - b[l]));
      b[l] = next;
    }
    if (delta < 1e-7) { iterations++; break; }
  }

  let aSum = 0, aCount = 0;
  for (let t = 0; t < T; t++) if (nT[t]) { aSum += a[t]; aCount++; }
  const aMean = aSum / aCount;

  const ss = new Float64Array(L);
  let ssAll = 0;
  for (let i = 0; i < N; i++) {
    const r = y[i] - a[task[i]] - b[lang[i]];
    ss[lang[i]] += r * r;
    ssAll += r * r;
  }
  const logEffect = new Float64Array(L);
  const typical = new Float64Array(L);
  const se = new Float64Array(L);
  for (let l = 0; l < L; l++) {
    logEffect[l] = n[l] ? b[l] + aMean : NaN;
    typical[l] = Math.exp(logEffect[l]);
    se[l] = n[l] >= 2 ? Math.sqrt(ss[l] / (n[l] - 1)) / Math.sqrt(n[l]) : NaN;
  }
  return { typical, logEffect, se, n, iterations, rmse: Math.sqrt(ssAll / N) };
}

export function naiveMean(data, values) {
  const { pairsByLang } = langIndex(data);
  return Float64Array.from(pairsByLang, (idx) => {
    let s = 0;
    for (const i of idx) s += values[i];
    return idx.length ? s / idx.length : NaN;
  });
}

export function ratioByLang(data, valuesA, valuesB) {
  const { pairsByLang } = langIndex(data);
  return Float64Array.from(pairsByLang, (idx) => {
    let sa = 0, sb = 0;
    for (const i of idx) { sa += valuesA[i]; sb += valuesB[i]; }
    return sb / sa;
  });
}

function median(xs) {
  const s = Float64Array.from(xs).sort();
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

export function taskDetail(data, langIdx, values) {
  const { task } = data.pairs;
  const byTask = new Map();
  const mine = [];
  for (let i = 0; i < task.length; i++) {
    const t = task[i];
    if (!byTask.has(t)) byTask.set(t, []);
    byTask.get(t).push(values[i]);
    if (data.pairs.lang[i] === langIdx) mine.push(i);
  }
  return mine
    .map((i) => {
      const taskMedian = median(byTask.get(task[i]));
      return { task: task[i], value: values[i], taskMedian, ratio: values[i] / taskMedian };
    })
    .sort((x, y) => x.ratio - y.ratio);
}
