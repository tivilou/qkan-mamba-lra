"use strict";

const ROOT_PATH = "five_ideas";
const DEFAULT_PATH =
  "five_ideas/phase9d_retrieval_rank_depth_replication/20260819T050703Z";
const SAFE_SEGMENT = /^[A-Za-z0-9._-]{1,128}$/;

const workspace = document.querySelector("#workspace");
const filePane = document.querySelector("#file-pane");
const itemList = document.querySelector("#item-list");
const filterInput = document.querySelector("#file-filter");
const breadcrumbs = document.querySelector("#breadcrumbs");
const statusPath = document.querySelector("#status-path");
const itemCount = document.querySelector("#item-count");
const treeRoot = document.querySelector("#tree-root");
const directoryPane = document.querySelector("#directory-pane");
const previewPane = document.querySelector("#preview-pane");
const previewTitle = document.querySelector("#preview-title");
const previewMeta = document.querySelector("#preview-meta");
const previewContent = document.querySelector("#preview-content");
const previewDownload = document.querySelector("#preview-download");
const dropOverlay = document.querySelector("#drop-overlay");
const dropTarget = document.querySelector("#drop-target");
const artifactInput = document.querySelector("#artifact-file");
const uploadFeedback = document.querySelector("#upload-feedback");
const uploadProgress = document.querySelector("#upload-progress");
const uploadStatus = document.querySelector("#upload-status");
const backButton = document.querySelector("#nav-back");
const forwardButton = document.querySelector("#nav-forward");
const upButton = document.querySelector("#nav-up");
const refreshButton = document.querySelector("#refresh-directory");

let currentPath = ROOT_PATH;
let currentPayload = { directories: [], files: [] };
let historyEntries = [];
let historyIndex = -1;
let dragDepth = 0;
const treeNodes = new Map();

function refreshIcons(root = document) {
  if (window.lucide) {
    window.lucide.createIcons({
      attrs: { "stroke-width": 1.8 },
      nameAttr: "data-lucide",
      root,
    });
  }
}

function encodedPath(path) {
  return path.split("/").map(encodeURIComponent).join("/");
}

function basename(path) {
  return path.split("/").at(-1);
}

function parentPath(path) {
  const segments = path.split("/");
  return segments.length > 1 ? segments.slice(0, -1).join("/") : null;
}

function formatBytes(value) {
  if (typeof value !== "number") return "";
  if (value < 1024) return String(value) + " B";
  const units = ["KB", "MB", "GB", "TB"];
  let size = value;
  let unit = -1;
  do {
    size /= 1024;
    unit += 1;
  } while (size >= 1024 && unit < units.length - 1);
  return size.toFixed(size >= 10 ? 1 : 2) + " " + units[unit];
}

function formatTime(value) {
  if (!value) return "";
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? value : date.toLocaleString();
}

function fileType(name) {
  const suffix = name.includes(".") ? name.split(".").at(-1).toLowerCase() : "";
  const labels = {
    csv: "CSV 文件",
    json: "JSON 文件",
    jsonl: "JSONL 文件",
    log: "日志文件",
    md: "Markdown 文件",
    txt: "文本文件",
    text: "文本文件",
    yaml: "YAML 文件",
    yml: "YAML 文件",
    zip: "ZIP 压缩包",
    gz: "Gzip 压缩包",
    tar: "TAR 归档",
    py: "Python 文件",
    sh: "Shell 脚本",
    pdf: "PDF 文件",
  };
  return labels[suffix] || (suffix ? suffix.toUpperCase() + " 文件" : "文件");
}

function fileIcon(name) {
  const suffix = name.includes(".") ? name.split(".").at(-1).toLowerCase() : "";
  if (["zip", "gz", "tar", "7z"].includes(suffix)) return "file-archive";
  if (["json", "jsonl", "yaml", "yml", "py", "sh"].includes(suffix)) {
    return "file-code-2";
  }
  if (suffix === "pdf") return "file-text";
  return "file";
}

function icon(name, className = "") {
  const wrapper = document.createElement("span");
  wrapper.className = "item-icon " + className;
  const element = document.createElement("i");
  element.dataset.lucide = name;
  element.setAttribute("aria-hidden", "true");
  wrapper.append(element);
  return wrapper;
}

function textCell(value, className = "") {
  const cell = document.createElement("td");
  cell.textContent = value;
  if (className) cell.className = className;
  return cell;
}

async function requestJson(url) {
  const response = await fetch(url, { credentials: "same-origin" });
  if (response.status === 401) {
    window.location.assign("/login");
    throw new Error("登录已过期");
  }
  let payload = {};
  try {
    payload = await response.json();
  } catch {
    payload = {};
  }
  if (!response.ok) {
    throw new Error(payload.error || "请求失败 (" + response.status + ")");
  }
  return payload;
}

async function fetchDirectory(path) {
  return requestJson("/api/files?path=" + encodeURIComponent(path));
}

function setLoading(message = "正在读取目录...") {
  itemList.replaceChildren();
  const row = document.createElement("tr");
  const cell = textCell(message, "empty-state");
  cell.colSpan = 6;
  row.append(cell);
  itemList.append(row);
}

function setHistory(path, mode) {
  if (mode === "back" || mode === "forward") return;
  if (historyEntries[historyIndex] === path) return;
  historyEntries = historyEntries.slice(0, historyIndex + 1);
  historyEntries.push(path);
  historyIndex = historyEntries.length - 1;
}

function updateNavigation() {
  backButton.disabled = historyIndex <= 0;
  forwardButton.disabled = historyIndex >= historyEntries.length - 1;
  upButton.disabled = currentPath === ROOT_PATH;
}

function updateUrl(path) {
  const url = new URL(window.location.href);
  url.searchParams.set("path", path);
  window.history.replaceState(null, "", url);
}

async function navigate(path, mode = "push") {
  if (!path || !path.startsWith(ROOT_PATH)) return;
  setLoading();
  closePreviewPane();
  refreshButton.classList.add("is-spinning");
  try {
    const payload = await fetchDirectory(path);
    currentPath = payload.path;
    currentPayload = payload;
    setHistory(currentPath, mode);
    updateUrl(currentPath);
    renderBreadcrumbs();
    renderItems();
    await revealTreePath(currentPath);
    statusPath.textContent = currentPath;
    dropTarget.textContent = "上传到 " + basename(currentPath);
  } catch (error) {
    setLoading(error.message);
  } finally {
    refreshButton.classList.remove("is-spinning");
    updateNavigation();
  }
}

function renderBreadcrumbs() {
  breadcrumbs.replaceChildren();
  const segments = currentPath.split("/");
  segments.forEach((segment, index) => {
    if (index > 0) {
      const separator = document.createElement("i");
      separator.dataset.lucide = "chevron-right";
      separator.className = "breadcrumb-separator";
      separator.setAttribute("aria-hidden", "true");
      breadcrumbs.append(separator);
    }
    const button = document.createElement("button");
    button.type = "button";
    button.className = "breadcrumb-button";
    button.textContent = segment;
    const path = segments.slice(0, index + 1).join("/");
    button.addEventListener("click", () => navigate(path));
    breadcrumbs.append(button);
  });
  refreshIcons(breadcrumbs);
}

function renderItems() {
  const query = filterInput.value.trim().toLowerCase();
  const directories = currentPayload.directories.filter((entry) =>
    entry.name.toLowerCase().includes(query)
  );
  const files = currentPayload.files.filter((entry) =>
    entry.name.toLowerCase().includes(query)
  );
  itemList.replaceChildren();

  if (!directories.length && !files.length) {
    const row = document.createElement("tr");
    const cell = textCell(query ? "没有匹配的项目" : "此文件夹为空", "empty-state");
    cell.colSpan = 6;
    row.append(cell);
    itemList.append(row);
    itemCount.textContent = "0 项";
    return;
  }

  for (const directory of directories) {
    const row = document.createElement("tr");
    row.className = "file-row directory-row";
    row.tabIndex = 0;
    row.dataset.path = directory.path;
    row.dataset.kind = "directory";

    const nameCell = document.createElement("td");
    nameCell.className = "name-cell";
    nameCell.append(icon("folder", "folder-icon"));
    const label = document.createElement("span");
    label.textContent = directory.name;
    nameCell.append(label);
    row.append(nameCell);
    row.append(textCell("文件夹", "muted-cell"));
    row.append(textCell("", "size-cell"));
    row.append(textCell(formatTime(directory.updated_at), "time-cell"));
    row.append(textCell("", "hash-cell"));
    row.append(document.createElement("td"));

    row.addEventListener("dblclick", () => navigate(directory.path));
    row.addEventListener("keydown", (event) => {
      if (event.key === "Enter") navigate(directory.path);
    });
    itemList.append(row);
  }

  for (const file of files) {
    const row = document.createElement("tr");
    row.className = "file-row";
    row.tabIndex = 0;
    row.dataset.path = file.path;
    row.dataset.kind = "file";

    const nameCell = document.createElement("td");
    nameCell.className = "name-cell";
    nameCell.append(icon(fileIcon(file.name)));
    const label = document.createElement("span");
    label.textContent = file.name;
    nameCell.append(label);
    row.append(nameCell);
    row.append(textCell(fileType(file.name), "muted-cell"));
    row.append(textCell(formatBytes(file.size_bytes), "size-cell"));
    row.append(textCell(formatTime(file.updated_at), "time-cell"));
    row.append(textCell(file.sha256 ? file.sha256.slice(0, 12) : "", "hash-cell"));

    const actionCell = document.createElement("td");
    actionCell.className = "row-actions";
    if (file.previewable) {
      const preview = document.createElement("button");
      preview.type = "button";
      preview.className = "icon-button subtle-action";
      preview.title = "预览";
      preview.setAttribute("aria-label", "预览 " + file.name);
      const previewIcon = document.createElement("i");
      previewIcon.dataset.lucide = "eye";
      preview.append(previewIcon);
      preview.addEventListener("click", (event) => {
        event.stopPropagation();
        showPreview(file);
      });
      actionCell.append(preview);
    }
    const download = document.createElement("a");
    download.className = "icon-button subtle-action";
    download.href = "/files/" + encodedPath(file.path);
    download.title = "下载";
    download.setAttribute("aria-label", "下载 " + file.name);
    const downloadIcon = document.createElement("i");
    downloadIcon.dataset.lucide = "download";
    download.append(downloadIcon);
    download.addEventListener("click", (event) => event.stopPropagation());
    actionCell.append(download);
    row.append(actionCell);

    row.addEventListener("click", () => selectRow(row));
    row.addEventListener("dblclick", () => {
      if (file.previewable) showPreview(file);
      else window.location.assign("/files/" + encodedPath(file.path));
    });
    row.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        if (file.previewable) showPreview(file);
        else window.location.assign("/files/" + encodedPath(file.path));
      }
    });
    itemList.append(row);
  }

  itemCount.textContent =
    String(currentPayload.directories.length + currentPayload.files.length) + " 项";
  refreshIcons(itemList);
}

function selectRow(row) {
  if (!row) return;
  itemList.querySelectorAll(".file-row.is-selected").forEach((entry) => {
    entry.classList.remove("is-selected");
  });
  row.classList.add("is-selected");
}

function createTreeNode(directory, depth, expanded = false) {
  const container = document.createElement("div");
  container.className = "tree-node";
  container.dataset.path = directory.path;
  container.setAttribute("role", "treeitem");
  container.setAttribute("aria-expanded", String(expanded));

  const row = document.createElement("div");
  row.className = "tree-row";
  row.style.setProperty("--tree-depth", String(depth));

  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.className = "tree-toggle";
  toggle.title = "展开";
  toggle.setAttribute("aria-label", "展开 " + directory.name);
  const toggleIcon = document.createElement("i");
  toggleIcon.dataset.lucide = expanded ? "chevron-down" : "chevron-right";
  toggle.append(toggleIcon);

  const open = document.createElement("button");
  open.type = "button";
  open.className = "tree-open";
  const folderIcon = document.createElement("i");
  folderIcon.dataset.lucide = expanded ? "folder-open" : "folder";
  const name = document.createElement("span");
  name.textContent = directory.name;
  open.append(folderIcon, name);

  const children = document.createElement("div");
  children.className = "tree-children";
  children.setAttribute("role", "group");
  children.hidden = !expanded;

  toggle.addEventListener("click", async (event) => {
    event.stopPropagation();
    await toggleTreeNode(container, directory.path, depth);
  });
  open.addEventListener("click", () => navigate(directory.path));
  row.append(toggle, open);
  container.append(row, children);
  treeNodes.set(directory.path, container);
  return container;
}

async function toggleTreeNode(node, path, depth, forceOpen = false) {
  const isOpen = node.getAttribute("aria-expanded") === "true";
  const shouldOpen = forceOpen || !isOpen;
  const children = node.querySelector(":scope > .tree-children");
  if (shouldOpen && node.dataset.loaded !== "true") {
    await loadTreeChildren(node, path, depth);
  }
  node.setAttribute("aria-expanded", String(shouldOpen));
  children.hidden = !shouldOpen;
  const toggleIcon = node.querySelector(":scope > .tree-row .tree-toggle i");
  const folderIcon = node.querySelector(":scope > .tree-row .tree-open i");
  if (toggleIcon) toggleIcon.dataset.lucide = shouldOpen ? "chevron-down" : "chevron-right";
  if (folderIcon) folderIcon.dataset.lucide = shouldOpen ? "folder-open" : "folder";
  refreshIcons(node.querySelector(":scope > .tree-row"));
}

async function loadTreeChildren(node, path, depth) {
  const children = node.querySelector(":scope > .tree-children");
  children.replaceChildren();
  try {
    const payload = await fetchDirectory(path);
    for (const directory of payload.directories) {
      children.append(createTreeNode(directory, depth + 1));
    }
    node.dataset.loaded = "true";
  } catch {
    node.dataset.loaded = "error";
  }
  refreshIcons(children);
}

async function revealTreePath(path) {
  if (!treeNodes.has(ROOT_PATH)) return;
  const segments = path.split("/");
  let partial = segments[0];
  for (let index = 0; index < segments.length; index += 1) {
    const node = treeNodes.get(partial);
    if (!node) break;
    if (index < segments.length - 1) {
      await toggleTreeNode(node, partial, index, true);
    }
    if (index < segments.length - 1) {
      partial += "/" + segments[index + 1];
    }
  }
  treeRoot.querySelectorAll(".tree-row.is-current").forEach((row) => {
    row.classList.remove("is-current");
  });
  const currentNode = treeNodes.get(path);
  currentNode?.querySelector(":scope > .tree-row")?.classList.add("is-current");
}

async function initializeTree() {
  treeRoot.replaceChildren();
  const rootNode = createTreeNode({ name: ROOT_PATH, path: ROOT_PATH }, 0, true);
  treeRoot.append(rootNode);
  await loadTreeChildren(rootNode, ROOT_PATH, 0);
  refreshIcons(treeRoot);
}

async function showPreview(file) {
  selectRow(itemList.querySelector('[data-path="' + CSS.escape(file.path) + '"]'));
  previewPane.hidden = false;
  workspace.classList.add("has-preview");
  previewTitle.textContent = file.name;
  previewMeta.textContent = "正在读取...";
  previewContent.textContent = "";
  previewDownload.href = "/files/" + encodedPath(file.path);
  try {
    const payload = await requestJson(
      "/api/preview?path=" + encodeURIComponent(file.path)
    );
    previewMeta.textContent =
      fileType(file.name) +
      " · " +
      formatBytes(payload.size_bytes) +
      (payload.truncated ? " · 仅显示前 256 KB" : "");
    previewContent.textContent = payload.content;
  } catch (error) {
    previewMeta.textContent = error.message;
  }
}

function closePreviewPane() {
  previewPane.hidden = true;
  workspace.classList.remove("has-preview");
}

function validateUploadFile(file) {
  const name = file.name;
  return (
    SAFE_SEGMENT.test(name) &&
    !name.startsWith(".") &&
    !name.endsWith(".upload.json")
  );
}

async function uploadFiles(fileListValue) {
  const files = Array.from(fileListValue);
  if (!files.length) return;
  const invalid = files.find((file) => !validateUploadFile(file));
  if (invalid) {
    showUploadStatus("文件名不符合规则：" + invalid.name, "error");
    return;
  }
  uploadFeedback.hidden = false;
  for (let index = 0; index < files.length; index += 1) {
    const file = files[index];
    const prefix = files.length > 1 ? String(index + 1) + "/" + files.length + " " : "";
    await uploadOne(file, prefix);
  }
  artifactInput.value = "";
  await navigate(currentPath, "refresh");
}

function uploadOne(file, prefix) {
  return new Promise((resolve) => {
    uploadProgress.value = 0;
    showUploadStatus(prefix + "正在上传 " + file.name, "");
    const request = new XMLHttpRequest();
    request.open("PUT", "/upload/" + encodedPath(currentPath + "/" + file.name));
    request.withCredentials = true;
    request.upload.addEventListener("progress", (event) => {
      if (event.lengthComputable) {
        uploadProgress.value = Math.round((event.loaded / event.total) * 100);
      }
    });
    request.addEventListener("load", () => {
      let payload = {};
      try {
        payload = JSON.parse(request.responseText);
      } catch {
        payload = {};
      }
      if (request.status === 200 || request.status === 201) {
        uploadProgress.value = 100;
        showUploadStatus(
          prefix +
            (request.status === 201
              ? "已上传 " + file.name
              : file.name + " 已存在"),
          "success"
        );
      } else if (request.status === 401) {
        window.location.assign("/login");
      } else {
        showUploadStatus(
          prefix + (payload.error || "上传失败 (" + request.status + ")"),
          "error"
        );
      }
      resolve();
    });
    request.addEventListener("error", () => {
      showUploadStatus(prefix + "上传连接失败：" + file.name, "error");
      resolve();
    });
    request.send(file);
  });
}

function showUploadStatus(message, state) {
  uploadStatus.textContent = message;
  uploadStatus.className = state ? "is-" + state : "";
}

function hideDropOverlay() {
  dragDepth = 0;
  dropOverlay.hidden = true;
  filePane.classList.remove("is-dragging");
}

document.querySelector("#upload-button").addEventListener("click", () => {
  artifactInput.click();
});
artifactInput.addEventListener("change", () => uploadFiles(artifactInput.files));
filterInput.addEventListener("input", renderItems);
refreshButton.addEventListener("click", () => navigate(currentPath, "refresh"));
upButton.addEventListener("click", () => {
  const parent = parentPath(currentPath);
  if (parent) navigate(parent);
});
backButton.addEventListener("click", () => {
  if (historyIndex <= 0) return;
  historyIndex -= 1;
  navigate(historyEntries[historyIndex], "back");
});
forwardButton.addEventListener("click", () => {
  if (historyIndex >= historyEntries.length - 1) return;
  historyIndex += 1;
  navigate(historyEntries[historyIndex], "forward");
});
document.querySelector("#close-preview").addEventListener("click", closePreviewPane);
document.querySelector("#toggle-sidebar").addEventListener("click", () => {
  directoryPane.classList.toggle("is-open");
});

filePane.addEventListener("dragenter", (event) => {
  event.preventDefault();
  dragDepth += 1;
  dropOverlay.hidden = false;
  filePane.classList.add("is-dragging");
});
filePane.addEventListener("dragover", (event) => {
  event.preventDefault();
  event.dataTransfer.dropEffect = "copy";
});
filePane.addEventListener("dragleave", (event) => {
  event.preventDefault();
  dragDepth -= 1;
  if (dragDepth <= 0) hideDropOverlay();
});
filePane.addEventListener("drop", (event) => {
  event.preventDefault();
  const files = event.dataTransfer.files;
  hideDropOverlay();
  uploadFiles(files);
});

window.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !previewPane.hidden) closePreviewPane();
});

async function start() {
  refreshIcons();
  await initializeTree();
  const requested = new URL(window.location.href).searchParams.get("path");
  const initial =
    requested && requested.startsWith(ROOT_PATH) ? requested : DEFAULT_PATH;
  await navigate(initial);
}

start();
