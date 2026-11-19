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
    id: 'perturbation',
    label: 'Perturbation',
    description: 'Average over five language axes and vision blur on the curated 12-task subset.',
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

function parseCsv(text) {
  const [header, ...lines] = text.trim().split(/\r?\n/);
  const columns = header.split(',');
  return lines.filter(Boolean).map((line) => {
    const values = line.split(',');
    const row = Object.fromEntries(columns.map((column, index) => [column, values[index]]));
    return {
      scene: row.scene,
      task: row.task,
      n: Number(row.n),
      sr: Number(row.SR),
      hsr: Number(row.HSR),
      cr: Number(row.CR),
    };
  });
}

async function fetchCsv(model, setting) {
  const path = `${model.id}/${setting.id}/${model.id}_${setting.id}.csv`;
  const url = `https://huggingface.co/datasets/${REPO}/resolve/${REVISION}/${path}?download=true`;
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}: ${url}`);
  return parseCsv(await response.text());
}

const results = {};
for (const model of MODELS) {
  results[model.id] = {};
  for (const setting of SETTINGS) {
    results[model.id][setting.id] = await fetchCsv(model, setting);
  }
}

const payload = {
  schema_version: 1,
  source: {
    label: 'RoboPro-Baselines on Hugging Face',
    url: `https://huggingface.co/datasets/${REPO}`,
    revision: REVISION,
    fetched_at: new Date().toISOString(),
  },
  metrics: {
    sr: { label: 'SR', direction: 'higher', description: 'Success rate' },
    hsr: { label: 'HSR', direction: 'higher', description: 'Hard success rate (success and no collision)' },
    cr: { label: 'CR', direction: 'lower', description: 'Collision rate' },
  },
  models: MODELS,
  settings: SETTINGS,
  scenes: SCENES,
  results,
};

await mkdir(dirname(outputPath), { recursive: true });
await writeFile(outputPath, `${JSON.stringify(payload, null, 2)}\n`, 'utf8');

const recordCount = Object.values(results)
  .flatMap((bySetting) => Object.values(bySetting))
  .reduce((total, rows) => total + rows.length, 0);
console.log(`Wrote ${outputPath} (${recordCount} per-task metric rows).`);
