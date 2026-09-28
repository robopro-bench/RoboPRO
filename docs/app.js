/* RoboPRO project page — vanilla JS for tab navigation and dropdown-driven viewers. */

(() => {
  const TAB_IDS = ['overview', 'tasks', 'collision', 'vision', 'language', 'leaderboard', 'rollouts'];
  let manifest = null;
  let leaderboardData = null;

  // ---------- Helpers ----------

  const $ = (sel, root = document) => root.querySelector(sel);
  const el = (tag, attrs = {}, children = []) => {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === 'class') node.className = v;
      else if (k === 'html') node.innerHTML = v;
      else if (v != null) node.setAttribute(k, v);
    }
    for (const c of [].concat(children || [])) {
      if (c == null) continue;
      node.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
    }
    return node;
  };

  const setVideo = (slotId, src, posterText) => {
    const slot = document.getElementById(slotId);
    if (!slot) return;
    slot.innerHTML = '';
    if (!src) {
      slot.appendChild(el('div', { class: 'video-fallback' },
        posterText || 'Rollout coming soon'));
      return;
    }
    const v = el('video', {
      controls: '', muted: '', loop: '', playsinline: '', preload: 'metadata',
      autoplay: ''
    });
    v.appendChild(el('source', { src, type: 'video/mp4' }));
    slot.appendChild(v);
  };

  const fillSelect = (select, options) => {
    select.innerHTML = '';
    for (const { value, label } of options) {
      const opt = el('option', { value }, label);
      select.appendChild(opt);
    }
  };

  // ---------- Tabs ----------

  const initTabs = () => {
    const links = document.querySelectorAll('.tab-nav a[data-tab]');
    const showTab = (id) => {
      if (!TAB_IDS.includes(id)) id = 'overview';
      links.forEach(a => a.classList.toggle('active', a.dataset.tab === id));
      TAB_IDS.forEach(t => {
        const panel = document.getElementById(t);
        if (panel) panel.classList.toggle('active', t === id);
      });
    };
    links.forEach(a => a.addEventListener('click', (e) => {
      e.preventDefault();
      const id = a.dataset.tab;
      history.replaceState(null, '', '#' + id);
      showTab(id);
      window.scrollTo({ top: document.querySelector('.tab-nav').offsetTop, behavior: 'smooth' });
    }));
    const initial = (location.hash || '').replace('#', '') || 'overview';
    showTab(initial);
  };

  // ---------- Task Gallery ----------

  const initTasks = () => {
    const sceneSel = $('#task-scene');
    const taskSel  = $('#task-slug');
    const scenes = Object.keys(manifest.scenes);

    fillSelect(sceneSel, scenes.map(k => ({ value: k, label: manifest.scenes[k].label })));

    const populateTasks = (scene) => {
      const list = manifest.tasks[scene] || [];
      fillSelect(taskSel, list.map(t => ({ value: t.slug, label: t.label + (t.kind === 'compositional' ? '  ·  comp.' : '') })));
    };

    const renderTask = () => {
      const scene = sceneSel.value;
      const slug  = taskSel.value;
      const list  = manifest.tasks[scene] || [];
      const t = list.find(x => x.slug === slug) || list[0];
      if (!t) return;
      $('#task-title').textContent = t.label;
      $('#task-desc').textContent = manifest.scenes[scene].blurb;
      $('#task-scene-label').textContent = manifest.scenes[scene].label;
      $('#task-kind').innerHTML = '';
      $('#task-kind').appendChild(el('span', {
        class: 'chip ' + (t.kind === 'compositional' ? 'compositional' : 'atomic')
      }, t.kind || 'atomic'));
      $('#task-slug-text').textContent = t.slug;
      setVideo('task-video-slot', t.video,
        'No rollout staged for ' + t.slug + ' yet');
    };

    sceneSel.addEventListener('change', () => { populateTasks(sceneSel.value); renderTask(); });
    taskSel.addEventListener('change', renderTask);

    sceneSel.value = scenes[0];
    populateTasks(scenes[0]);
    renderTask();
  };

  // ---------- Collision ----------

  const initCollision = () => {
    const sel = $('#collision-density');
    const levels = manifest.collision.levels;
    fillSelect(sel, levels.map(l => ({ value: l.key, label: l.label })));

    const render = () => {
      const lvl = levels.find(l => l.key === sel.value) || levels[0];
      $('#collision-title').textContent = lvl.label;
      $('#collision-count').textContent = String(lvl.obstacles);
      $('#collision-task').textContent = manifest.collision.task;
      $('#collision-scene').textContent =
        manifest.scenes[manifest.collision.scene]?.label || manifest.collision.scene;
      setVideo('collision-video-slot', lvl.video);
    };

    sel.addEventListener('change', render);
    sel.value = 'd10';
    render();
  };

  // ---------- Vision ----------

  const initVision = () => {
    const sel = $('#vision-axis');
    const axes = manifest.vision.axes;
    fillSelect(sel, axes.map(a => ({ value: a.key, label: a.label })));

    const render = () => {
      const a = axes.find(x => x.key === sel.value) || axes[0];
      $('#vision-title').textContent = a.label;
      $('#vision-blurb').textContent = a.blurb;
      setVideo('vision-video-slot', a.video);
    };

    sel.addEventListener('change', render);
    sel.value = 'combined';
    render();
  };

  // ---------- Language ----------

  const initLanguage = () => {
    const sel = $('#language-axis');
    const axes = manifest.language.axes;
    fillSelect(sel, axes.map(a => ({ value: a.key, label: a.label })));

    const render = () => {
      const a = axes.find(x => x.key === sel.value) || axes[0];
      $('#language-title').textContent = a.label;
      $('#language-sample').textContent = a.sample || '';
      setVideo('language-video-slot', a.video);
    };

    sel.addEventListener('change', render);
    sel.value = axes[0].key;
    render();
  };

  // ---------- Leaderboard ----------

  const initLeaderboard = () => {
    if (!leaderboardData) {
      $('#lb-ranking-note').textContent = 'Leaderboard data could not be loaded. Please refresh the page.';
      return;
    }

    const state = { setting: 'clean', metric: 'sr', scene: 'all', task: 'all', view: 'all' };
    const metricSelect = $('#lb-metric');
    const sceneSelect = $('#lb-scene');
    const taskSelect = $('#lb-task');
    const settingButtons = [...document.querySelectorAll('[data-lb-setting]')];
    const viewButtons = [...document.querySelectorAll('[data-lb-view]')];
    const sceneOrder = new Map(leaderboardData.scenes.map((scene, index) => [scene.id, index]));
    const taskLabel = (scene, task) =>
      manifest.tasks?.[scene]?.find((entry) => entry.slug === task)?.label ||
      task.replace(/_/g, ' ').replace(/\b\w/g, (char) => char.toUpperCase());
    const sceneLabel = (scene) =>
      leaderboardData.scenes.find((entry) => entry.id === scene)?.label || scene;
    const metric = () => leaderboardData.metrics[state.metric];
    const setting = () => leaderboardData.settings.find((entry) => entry.id === state.setting);
    const metricLabel = () => `${metric().label} ${metric().direction === 'higher' ? '↑' : '↓'}`;
    const percent = (value) => Number.isFinite(value) ? `${value.toFixed(1)}%` : '—';
    const recordsFor = (modelId) => (leaderboardData.results[modelId]?.[state.setting] || [])
      .filter((row) => state.scene === 'all' || row.scene === state.scene)
      .filter((row) => state.task === 'all' || `${row.scene}/${row.task}` === state.task);
    const aggregate = (records) => {
      const n = records.reduce((sum, row) => sum + row.n, 0);
      const weighted = (key) => n
        ? records.reduce((sum, row) => sum + row[key] * row.n, 0) / n
        : null;
      return { n, tasks: records.length, sr: weighted('sr'), hsr: weighted('hsr'), cr: weighted('cr') };
    };
    const taskOptions = () => {
      const all = new Map();
      for (const model of leaderboardData.models) {
        for (const row of leaderboardData.results[model.id]?.[state.setting] || []) {
          if (state.scene !== 'all' && row.scene !== state.scene) continue;
          all.set(`${row.scene}/${row.task}`, row);
        }
      }
      return [...all.entries()]
        .sort(([, a], [, b]) => (sceneOrder.get(a.scene) - sceneOrder.get(b.scene)) ||
          taskLabel(a.scene, a.task).localeCompare(taskLabel(b.scene, b.task)))
        .map(([value, row]) => ({ value, label: `${sceneLabel(row.scene)} · ${taskLabel(row.scene, row.task)}` }));
    };
    const ranking = () => leaderboardData.models
      .map((model) => ({ model, scores: aggregate(recordsFor(model.id)) }))
      .filter((entry) => entry.scores.n)
      .sort((a, b) => {
        const delta = a.scores[state.metric] - b.scores[state.metric];
        return metric().direction === 'higher' ? -delta : delta;
      });
    const availableTasks = () => {
      const tasks = new Map();
      for (const model of leaderboardData.models) {
        for (const row of recordsFor(model.id)) {
          tasks.set(`${row.scene}/${row.task}`, { scene: row.scene, task: row.task });
        }
      }
      return [...tasks.values()];
    };
    const taskResultRows = (ranked) => {
      const rows = availableTasks().map(({ scene, task }) => {
        const byModel = ranked.map(({ model }) =>
          (leaderboardData.results[model.id]?.[state.setting] || [])
            .find((row) => row.scene === scene && row.task === task));
        const values = byModel.filter(Boolean).map((row) => row[state.metric]);
        return {
          scene,
          task,
          byModel,
          score: values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null,
        };
      });
      if (state.view === 'hardest' && state.task === 'all') {
        rows.sort((a, b) => metric().direction === 'higher' ? a.score - b.score : b.score - a.score);
        return rows.slice(0, 10);
      }
      return rows.sort((a, b) => (sceneOrder.get(a.scene) - sceneOrder.get(b.scene)) ||
        taskLabel(a.scene, a.task).localeCompare(taskLabel(b.scene, b.task)));
    };

    const render = () => {
      const options = taskOptions();
      if (state.task !== 'all' && !options.some((option) => option.value === state.task)) state.task = 'all';
      fillSelect(taskSelect, [{ value: 'all', label: 'All available tasks' }, ...options]);
      taskSelect.value = state.task;

      const currentSetting = setting();
      $('#lb-setting-title').textContent = currentSetting.label;
      $('#lb-setting-description').textContent = currentSetting.description;
      settingButtons.forEach((button) => {
        const active = button.dataset.lbSetting === state.setting;
        button.classList.toggle('active', active);
        button.setAttribute('aria-pressed', String(active));
      });
      viewButtons.forEach((button) => {
        const active = button.dataset.lbView === state.view;
        button.classList.toggle('active', active);
        button.setAttribute('aria-pressed', String(active));
      });

      const ranked = ranking();
      const exemplar = ranked[0]?.scores || { n: 0, tasks: 0 };
      $('#lb-task-count').textContent = String(exemplar.tasks);
      $('#lb-episode-count').textContent = String(exemplar.n);
      $('#lb-method-count').textContent = String(ranked.length);
      $('#lb-update-date').textContent = new Intl.DateTimeFormat(undefined, {
        year: 'numeric', month: 'short', day: 'numeric'
      }).format(new Date(leaderboardData.source.fetched_at));

      const scope = [
        currentSetting.label,
        state.scene === 'all' ? 'all scenes' : sceneLabel(state.scene),
        state.task === 'all' ? `${exemplar.tasks} tasks` : taskLabel(...state.task.split('/')),
      ].join(' · ');
      $('#lb-ranking-heading').textContent = `Overall Ranking — ${scope}`;
      $('#lb-ranking-note').textContent =
        `Ranked by ${metricLabel()}. Scores are micro-averaged over the selected evaluation episodes; CR is lower-is-better.`;

      const rankingBody = $('#lb-ranking');
      rankingBody.innerHTML = '';
      ranked.forEach(({ model, scores }, index) => {
        const row = el('tr', {}, [
          el('td', { class: 'num rank' }, String(index + 1)),
          el('td', { class: 'method' }, model.name),
          el('td', { class: `num${state.metric === 'sr' ? ' ranked-metric' : ''}` }, percent(scores.sr)),
          el('td', { class: `num${state.metric === 'hsr' ? ' ranked-metric' : ''}` }, percent(scores.hsr)),
          el('td', { class: `num${state.metric === 'cr' ? ' ranked-metric' : ''}` }, percent(scores.cr)),
          el('td', { class: 'num' }, String(scores.n)),
          el('td', { class: 'num' }, String(scores.tasks)),
        ]);
        rankingBody.appendChild(row);
      });

      const perTask = taskResultRows(ranked);
      $('#lb-task-heading').textContent = state.task === 'all'
        ? `Per-task Results — ${metricLabel()}`
        : `Task Result — ${taskLabel(...state.task.split('/'))}`;
      $('#lb-task-note').textContent = state.view === 'hardest' && state.task === 'all'
        ? `Showing the ten most difficult selected tasks by mean ${metric().label} across the released methods. Trial counts follow the method-column order.`
        : `Each cell is the selected metric for one task. Trial counts follow the method-column order; a dash means the source has no result for that cell.`;
      const taskHead = $('#lb-task-head');
      taskHead.innerHTML = '';
      taskHead.appendChild(el('tr', {}, [
        el('th', {}, 'Task'),
        el('th', {}, 'Scene'),
        el('th', { class: 'num' }, 'Trials'),
        ...ranked.map(({ model }) => el('th', { class: 'num' }, model.name)),
      ]));
      const taskBody = $('#lb-task-results');
      taskBody.innerHTML = '';
      perTask.forEach(({ scene, task, byModel }) => {
        const values = byModel.map((row) => row?.[state.metric]).filter(Number.isFinite);
        const best = values.length ? (metric().direction === 'higher' ? Math.max(...values) : Math.min(...values)) : null;
        const trialCounts = byModel.map((row) => row?.n ?? '—');
        taskBody.appendChild(el('tr', {}, [
          el('td', { class: 'task-name' }, taskLabel(scene, task)),
          el('td', { class: 'scene-name' }, sceneLabel(scene)),
          el('td', { class: 'num' }, [...new Set(trialCounts)].join(' / ')),
          ...byModel.map((row) => el('td', {
            class: `num${row && row[state.metric] === best ? ' best' : ''}`
          }, row ? percent(row[state.metric]) : '—')),
        ]));
      });
    };

    fillSelect(metricSelect, Object.entries(leaderboardData.metrics).map(([id, entry]) => ({
      value: id,
      label: `${entry.label} · ${entry.description} (${entry.direction === 'higher' ? 'higher is better' : 'lower is better'})`,
    })));
    fillSelect(sceneSelect, [
      { value: 'all', label: 'All scenes' },
      ...leaderboardData.scenes.map((scene) => ({ value: scene.id, label: scene.label })),
    ]);
    metricSelect.value = state.metric;
    sceneSelect.value = state.scene;
    settingButtons.forEach((button) => button.addEventListener('click', () => {
      state.setting = button.dataset.lbSetting;
      state.task = 'all';
      render();
    }));
    viewButtons.forEach((button) => button.addEventListener('click', () => {
      state.view = button.dataset.lbView;
      render();
    }));
    metricSelect.addEventListener('change', () => { state.metric = metricSelect.value; render(); });
    sceneSelect.addEventListener('change', () => { state.scene = sceneSelect.value; state.task = 'all'; render(); });
    taskSelect.addEventListener('change', () => { state.task = taskSelect.value; render(); });
    render();
  };

  // ---------- Rollouts ----------

  const initRollouts = () => {
    const grid = $('#rollouts-grid');
    grid.innerHTML = '';
    for (const r of manifest.rollouts) {
      const v = el('video', {
        controls: '', muted: '', loop: '', playsinline: '', preload: 'metadata'
      });
      v.appendChild(el('source', { src: r.video, type: 'video/mp4' }));
      const fig = el('figure', { class: 'rollout' }, [
        v,
        el('figcaption', {}, [
          el('span', { class: 'policy' }, r.policy),
          el('span', { class: 'task' }, r.task),
          el('span', { class: 'scene' }, r.scene)
        ])
      ]);
      grid.appendChild(fig);
    }
  };

  // ---------- Boot ----------

  const boot = async () => {
    initTabs();
    try {
      const res = await fetch('manifest.json?v=6', { cache: 'no-cache' });
      manifest = await res.json();
    } catch (e) {
      console.error('Failed to load manifest.json:', e);
      return;
    }
    try {
      const res = await fetch('leaderboard-data.json?v=1', { cache: 'no-cache' });
      leaderboardData = await res.json();
    } catch (e) {
      console.error('Failed to load leaderboard data:', e);
    }
    initTasks();
    initCollision();
    initVision();
    initLanguage();
    initLeaderboard();
    initRollouts();
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
