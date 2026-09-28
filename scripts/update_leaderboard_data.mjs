#!/usr/bin/env node
/**
 * Snapshot RoboPRO baseline metrics for the static project website.
 *
 * Source: https://huggingface.co/datasets/JackLiu0406/RoboPro-Baselines
 * Run:   node scripts/update_leaderboard_data.mjs
 */

import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const REPO = 'JackLiu0406/RoboPro-Baselines';
const REVISION = 'main';
const MODELS = [
  { id: 'pi05', name: 'π0.5' },
  { id: 'pi0', name: 'π0' },
  { id: 'xvla', name: 'X-VLA' },
];
const SETTINGS = [
  {
    id: 'clean',
    label: 'Clean',
    description: 'Obstacle-free evaluation across the released RoboPRO baseline suite.',
  },
  {
    id: 'clutter',
    label: 'Clutter',
    description: 'Average over obstacle densities d6–d15.',
  },
  {
    id: 'perturbation_clean',
    label: 'Perturbation · Clean',
    description: 'Average over all 13 perturbation axes (language, distractor, vision, object OOD) on the curated 12-task subset, obstacle-free scenes.',
    source: 'perturbation',
    condition: 'clean',
  },
  {
    id: 'perturbation_clutter',
    label: 'Perturbation · Clutter',
    description: 'Average over all 13 perturbation axes on the curated 12-task subset, clutter densities d6–d15. Not comparable to the Clutter setting, which uses different clutter configs.',
    source: 'perturbation',
    condition: 'clutter',
  },
];
const SCENES = [
  { id: 'office', label: 'Office' },
  { id: 'study', label: 'Study' },
  { id: 'kitchens', label: 'Kitchen-S' },
  { id: 'kitchenl', label: 'Kitchen-L' },
];

const here = dirname(fileURLToPath(import.meta.url));
const outputPath = resolve(here, '..', 'docs', 'leaderboard-data.json');

function readCsv(text) {
  const [header, ...lines] = text.trim().split(/\r?\n/);
  const columns = header.split(',');
  return lines.filter(Boolean).map((line) => {
    const values = line.split(',');
    return Object.fromEntries(columns.map((column, index) => [column, values[index]]));
  });
}

// <model>/<setting>/<model>_<setting>.csv: one row per task.
function parseTaskCsv(text) {
  return readCsv(text).map((row) => ({
    scene: row.scene,
    task: row.task,
    n: Number(row.n),
    sr: Number(row.SR),
    hsr: Number(row.HSR),
    cr: Number(row.CR),
  }));
}

// <model>/perturbation/<model>_perturbation.csv: one row per axis/scene/task/condition.
// Like the dataset's own summary tables, only complete cells count; each task is
// then micro-averaged over episodes across the 13 axes.
function parsePerturbationCsv(text, condition) {
  const byTask = new Map();
  for (const row of readCsv(text)) {
    if (row.condition !== condition || row.complete !== 'True') continue;
    const key = `${row.scene}/${row.task}`;
    const acc = byTask.get(key) || { scene: row.scene, task: row.task, n: 0, sr: 0, hsr: 0, cr: 0 };
    const n = Number(row.episodes);
    acc.n += n;
    for (const metric of ['sr', 'hsr', 'cr']) acc[metric] += Number(row[metric]) * n;
    byTask.set(key, acc);
  }
  return [...byTask.values()].map((acc) => ({
    ...acc,
    sr: acc.sr / acc.n,
    hsr: acc.hsr / acc.n,
    cr: acc.cr / acc.n,
  }));
}

async function fetchText(path) {
  const url = `https://huggingface.co/datasets/${REPO}/resolve/${REVISION}/${path}?download=true`;
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}: ${url}`);
  return response.text();
}

async function fetchRows(model, setting) {
  const source = setting.source || setting.id;
  const text = await fetchText(`${model.id}/${source}/${model.id}_${source}.csv`);
  return setting.condition ? parsePerturbationCsv(text, setting.condition) : parseTaskCsv(text);
}

const info = await (await fetch(`https://huggingface.co/api/datasets/${REPO}/revision/${REVISION}`)).json();

const results = {};
for (const model of MODELS) {
  results[model.id] = {};
  for (const setting of SETTINGS) {
    results[model.id][setting.id] = await fetchRows(model, setting);
  }
}

const payload = {
  schema_version: 1,
  source: {
    label: 'RoboPro-Baselines on Hugging Face',
    url: `https://huggingface.co/datasets/${REPO}`,
    revision: info.sha,
    source_updated_at: info.lastModified,
    fetched_at: new Date().toISOString(),
  },
  metrics: {
    sr: { label: 'SR', direction: 'higher', description: 'Success rate' },
    hsr: { label: 'HSR', direction: 'higher', description: 'Hard success rate (success and no collision)' },
    cr: { label: 'CR', direction: 'lower', description: 'Collision rate' },
  },
  models: MODELS,
  settings: SETTINGS.map(({ id, label, description }) => ({ id, label, description })),
  scenes: SCENES,
  results,
};

await mkdir(dirname(outputPath), { recursive: true });
await writeFile(outputPath, `${JSON.stringify(payload, null, 2)}\n`, 'utf8');

const recordCount = Object.values(results)
  .flatMap((bySetting) => Object.values(bySetting))
  .reduce((total, rows) => total + rows.length, 0);
console.log(`Wrote ${outputPath} (${recordCount} per-task metric rows).`);
