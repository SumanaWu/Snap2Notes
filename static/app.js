const $ = id => document.getElementById(id);
const svgNS = 'http://www.w3.org/2000/svg';
let noteId;
let note;
let files = [];
let selection = null;
let firstPoint;
let selectedBlock;
let timer;
let lastSavedMarkdown = '';
let lastSavedConfirmed = false;

async function request(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(typeof body.detail === 'string' ? body.detail : response.statusText);
  }
  return response;
}

const json = (method, data) => ({
  method, headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data),
});
const message = text => { $('status').textContent = text; };
const report = action => async (...args) => {
  try { await action(...args); } catch (error) { message(error.message); }
};
const endpoint = suffix => `/api/notes/${noteId}${suffix}`;
const pageEndpoint = suffix => endpoint(`/pages/${$('page').value}${suffix}`);

function encode(item) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve({name: item.file.name, surface: item.surface, data: reader.result.split(',')[1]});
    reader.onerror = reject;
    reader.readAsDataURL(item.file);
  });
}

function renderFiles() {
  const list = $('selected');
  list.replaceChildren();
  if (!files.length) list.textContent = '尚未选择照片';
  files.forEach((item, index) => {
    const row = document.createElement('div');
    row.className = 'file-row';
    const name = document.createElement('span');
    name.textContent = item.file.name;
    const surface = document.createElement('select');
    surface.setAttribute('aria-label', `${item.file.name} 的表面类型`);
    surface.append(new Option('纸张', 'paper'), new Option('白板', 'whiteboard'), new Option('黑板', 'blackboard'));
    surface.value = item.surface;
    surface.onchange = () => { item.surface = surface.value; };
    const up = document.createElement('button');
    up.textContent = '上移';
    up.className = 'secondary';
    up.disabled = index === 0;
    up.onclick = () => {
      [files[index - 1], files[index]] = [files[index], files[index - 1]];
      renderFiles();
    };
    row.append(name, surface, up);
    list.append(row);
  });
  $('start').disabled = !files.length;
}

$('files').onchange = () => {
  files = [...$('files').files].map(file => ({file, surface: 'paper'}));
  renderFiles();
};
$('start').onclick = report(async () => {
  if (files.length > 12 || files.some(item => item.file.size > 20 * 1024 * 1024)) {
    throw new Error('每次最多 12 张，每张不超过 20 MB。');
  }
  $('start').disabled = true;
  message('正在上传照片…');
  try {
    if (noteId && !$('workspace').hidden && isDirty()) await save();
    const images = await Promise.all(files.map(encode));
    const result = await request('/api/notes', json('POST', {images, same_board: $('same-board').checked}));
    await openNote((await result.json()).id);
    await loadHistory();
  } finally {
    $('start').disabled = !files.length;
  }
});

async function loadHistory() {
  const history = await (await request('/api/notes')).json();
  $('history').replaceChildren(new Option('打开历史笔记', ''),
    ...history.map(item => new Option(item.name, item.id)));
  $('history').value = noteId || '';
}

async function openNote(id, page = 0) {
  clearTimeout(timer);
  noteId = id;
  $('excerpts').replaceChildren();
  const result = await (await request(endpoint(''))).json();
  if (id !== noteId) return;
  note = result;
  const active = ['queued', 'processing'].includes(note.status);
  $('workspace').hidden = active;
  if (active) {
    const finished = note.pages.filter(item => Object.hasOwn(item, 'error')).length;
    message(`正在识别 · ${finished}/${note.pages.length}`);
    timer = setTimeout(report(() => openNote(id, page)), 1400);
    return;
  }
  const notices = note.pages.flatMap(item => [item.error, ...(item.warnings || [])].filter(Boolean).map(text => `${item.name} · ${text}`));
  message([note.status === 'interrupted' ? '识别已中断，可以重试本页。' : '请核对识别结果。', note.warning, ...notices].filter(Boolean).join('\n'));
  $('page').replaceChildren(...note.pages.map((item, index) => new Option(item.name, index)));
  $('page').value = page;
  $('markdown').value = note.markdown;
  $('confirmed').checked = Boolean(note.confirmed);
  lastSavedMarkdown = note.markdown;
  lastSavedConfirmed = Boolean(note.confirmed);
  previewMath();
  showPage();
}

function showPage(highlight = null) {
  selection = null;
  selectedBlock = null;
  const page = note.pages[Number($('page').value)];
  $('image').src = pageEndpoint('');
  $('boxes').setAttribute('viewBox', `0 0 ${page.width} ${page.height}`);
  $('boxes').replaceChildren();
  $('block-info').textContent = '点击识别框查看来源和内容。';
  for (const block of page.blocks) {
    const shape = document.createElementNS(svgNS, 'polygon');
    shape.setAttribute('points', block.polygon.map(point => point.join(',')).join(' '));
    if (block.polygon === highlight) shape.classList.add('highlight');
    shape.onclick = () => {
      if (selection) return;
      const status = block.duplicate_of ? ' · 已合并的重复内容' : block.possible_revision_of ? ' · 疑似改写，已保留' : '';
      selectedBlock = block;
      const info = $('block-info');
      info.textContent = (block.text || block.kind || '文字') + status;
      const search = block.text;
      const offset = search ? $('markdown').value.indexOf(search) : -1;
      if (offset >= 0) $('markdown').setSelectionRange(offset, offset + search.length);
    };
    $('boxes').append(shape);
  }
}

function imagePoint(event) {
  const svg = $('boxes');
  const point = new DOMPoint(event.clientX, event.clientY).matrixTransform(svg.getScreenCTM().inverse());
  const page = note.pages[Number($('page').value)];
  return [Math.max(0, Math.min(page.width - 1, Math.round(point.x))),
    Math.max(0, Math.min(page.height - 1, Math.round(point.y)))];
}

$('corners').onclick = () => {
  selection = {corners: []};
  $('block-info').textContent = '依次点击照片中板面或纸页的四个角。';
};
$('region').onclick = () => {
  selection = {kind: $('region-kind').value};
  firstPoint = null;
  $('block-info').textContent = '点击选区左上角和右下角。';
};
$('boxes').onclick = report(async event => {
  if (!selection) return;
  const point = imagePoint(event);
  if (selection.corners) {
    selection.corners.push(point);
    $('block-info').textContent = `已标记 ${selection.corners.length}/4 个角。`;
    if (selection.corners.length === 4) {
      const corners = selection.corners;
      selection = null;
      await retry(corners);
    }
    return;
  }
  if (!firstPoint) {
    firstPoint = point;
    return;
  }
  const [x, y] = firstPoint;
  const [right, bottom] = point;
  const kind = selection.kind;
  selection = null;
  const box = [Math.min(x, right), Math.min(y, bottom), Math.max(x, right) + 1, Math.max(y, bottom) + 1];
  const result = await (await request(pageEndpoint('/region'), json('POST', {kind, box}))).json();
  note.pages[Number($('page').value)].blocks.push(result);
  insert(result.markdown);
  await save();
  showPage();
});

function previewMath() {
  const source = $('markdown').value;
  const expressions = [...source.matchAll(/\$\$([\s\S]+?)\$\$|\$([^$\n]+)\$/g)].slice(0, 12);
  $('math-preview').replaceChildren(...expressions.map(match => {
    const preview = document.createElement('span');
    const expression = (match[1] || match[2]).trim();
    if (window.katex) window.katex.render(expression, preview, {throwOnError: false, trust: false});
    else preview.textContent = expression;
    return preview;
  }));
}

function insert(text) {
  if (!text) return;
  const editor = $('markdown');
  editor.setRangeText(`\n\n${text}\n\n`, editor.selectionStart, editor.selectionEnd, 'end');
  $('confirmed').checked = false;
  previewMath();
}

async function save() {
  // 记录请求提交时的内容，避免等待期间的新编辑被视为已保存。
  const markdown = $('markdown').value;
  const confirmed = $('confirmed').checked;
  await request(endpoint(''), json('PUT', {markdown, confirmed}));
  lastSavedMarkdown = markdown;
  lastSavedConfirmed = confirmed;
  message('修改已保存。');
}
function isDirty() {
  return $('markdown').value !== lastSavedMarkdown || $('confirmed').checked !== lastSavedConfirmed;
}
async function retry(corners = null) {
  const index = Number($('page').value);
  if ($('markdown').value !== note.markdown) await save();
  await request(pageEndpoint('/retry'), json('POST', {corners}));
  await openNote(noteId, index);
}

$('markdown').oninput = () => { $('confirmed').checked = false; previewMath(); };
$('history').onchange = report(async () => {
  const selected = $('history').value;
  if (!selected || selected === noteId) return;
  try {
    // 切换历史笔记前先保存当前编辑，保存失败时继续留在当前笔记。
    if (noteId && !$('workspace').hidden && isDirty()) await save();
    await openNote(selected);
  } catch (error) {
    $('history').value = noteId || '';
    throw error;
  }
});
window.addEventListener('beforeunload', event => {
  if (noteId && !$('workspace').hidden && isDirty()) {
    event.preventDefault();
    event.returnValue = '';
  }
});
$('page').onchange = () => showPage();
$('save').onclick = report(save);
$('retry').onclick = report(() => retry());
$('draft').onclick = () => insert(note.draft);
$('summary').onclick = report(async () => {
  await save();
  const excerpts = await (await request(endpoint('/summary'))).json();
  $('excerpts').replaceChildren(...excerpts.map(item => {
    const button = document.createElement('button');
    button.className = 'secondary';
    button.textContent = item.text;
    button.onclick = () => {
      $('page').value = item.page;
      showPage(note.pages[item.page].blocks.find(block => JSON.stringify(block.polygon) === JSON.stringify(item.polygon))?.polygon);
    };
    return button;
  }));
  if (!excerpts.length) $('excerpts').textContent = '暂无可追溯的原文摘录。';
});
$('export').onclick = report(async () => {
  await save();
  const response = await request(endpoint(`/export/${$('format').value}`));
  const url = URL.createObjectURL(await response.blob());
  const link = document.createElement('a');
  link.href = url;
  link.download = response.headers.get('Content-Disposition').match(/filename="([^"]+)"/)[1];
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
report(loadHistory)();
