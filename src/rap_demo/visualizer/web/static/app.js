/**
 * RAP Real-Time Visualizer Client
 * Handles SSE streaming, pan/zoom SVG tree rendering, Blocksworld tabletop,
 * trajectory scrubbing, structured node inspection, and minimal theme switching.
 */

class VisualizerApp {
  constructor() {
    this.treeData = null;
    this.stateData = null;
    this.status = "idle";
    this.iteration = 0;
    this.maxIterations = 50;
    this.selectedNode = null;
    this.bestPathIds = new Set();
    this.bestPathNodes = [];
    this.autoFollow = true;
    this.inspectorMode = "overview";
    this.isStepping = false;
    this.stepStartTime = 0;
    this.stepTicker = null;
    this.syncTimer = null;
    this.eventSource = null;
    this.liveStream = {
      active: false,
      iteration: 0,
      phase: "",
      callId: null,
      reasoning: "",
      content: "",
    };
    this.liveAutoScroll = true;
    this.liveElements = null;

    // Pan & Zoom state
    this.panX = 0;
    this.panY = 0;
    this.zoom = 1;
    this.hasManualViewport = false;
    this.isPanning = false;
    this.panStart = { x: 0, y: 0 };
    this.treeBounds = { width: 400, height: 300 };
    this.nodeElements = new Map();

    this.initElements();
    this.initTheme();
    this.bindEvents();
    this.connectSSE();
    this.fetchInitialData();
  }

  initElements() {
    this.modeBadge = document.getElementById("mode-badge");
    this.sseDot = document.getElementById("sse-dot");
    this.statusDot = document.getElementById("status-dot");
    this.statusText = document.getElementById("status-text");
    this.iterationCounter = document.getElementById("iteration-counter");
    this.progressFill = document.getElementById("progress-fill");

    this.btnStep = document.getElementById("btn-step");
    this.btnPlay = document.getElementById("btn-play");
    this.btnPlayLabel = document.getElementById("btn-play-label");
    this.btnPause = document.getElementById("btn-pause");
    this.btnReset = document.getElementById("btn-reset");
    this.speedSlider = document.getElementById("speed-slider");
    this.speedValue = document.getElementById("speed-value");

    this.btnTheme = document.getElementById("btn-theme");
    this.iconSun = document.getElementById("icon-sun");
    this.iconMoon = document.getElementById("icon-moon");

    this.treeContainer = document.getElementById("tree-container");
    this.treeSvg = document.getElementById("tree-svg");
    this.treeViewport = document.getElementById("tree-viewport");
    this.treeSummary = document.getElementById("tree-summary");
    this.treeTooltip = document.getElementById("tree-tooltip");
    this.btnZoomIn = document.getElementById("btn-zoom-in");
    this.btnZoomOut = document.getElementById("btn-zoom-out");
    this.btnFitTree = document.getElementById("btn-fit-tree");
    this.zoomReadout = document.getElementById("zoom-readout");

    this.goalStrip = document.getElementById("goal-strip");
    this.goalText = document.getElementById("goal-text");
    this.holdingPill = document.getElementById("holding-pill");
    this.stacksArea = document.getElementById("stacks-area");
    this.heldBlockSlot = document.getElementById("held-block-slot");

    this.trajectoryStrip = document.getElementById("trajectory-strip");
    this.btnFollowBest = document.getElementById("btn-follow-best");
    this.tabOverview = document.getElementById("tab-overview");
    this.tabRaw = document.getElementById("tab-raw");
    this.thoughtModeTag = document.getElementById("thought-mode-tag");
    this.thoughtDisplay = document.getElementById("thought-display");
    this.nodeIdBadge = document.getElementById("node-id-badge");
    this.terminalBadge = document.getElementById("terminal-badge");
    this.metricsOverview = document.getElementById("metrics-overview");
    this.nodeInspector = document.getElementById("node-inspector");
    this.stateOverview = document.getElementById("state-overview");
    this.stateInspector = document.getElementById("state-inspector");
    this.toastContainer = document.getElementById("toast-container");
  }

  initTheme() {
    const saved = localStorage.getItem("rap-theme") || "dark";
    this.applyTheme(saved);
  }

  applyTheme(theme) {
    const isLight = theme === "light";
    if (isLight) {
      document.documentElement.setAttribute("data-theme", "light");
    } else {
      document.documentElement.removeAttribute("data-theme");
    }
    if (this.iconSun && this.iconMoon) {
      this.iconSun.hidden = !isLight;
      this.iconMoon.hidden = isLight;
    }
    localStorage.setItem("rap-theme", isLight ? "light" : "dark");
  }

  bindEvents() {
    if (this.btnStep) this.btnStep.addEventListener("click", () => this.postAction("/api/search/step"));
    if (this.btnPlay) {
      this.btnPlay.addEventListener("click", () => {
        const endpoint = this.status === "paused" ? "/api/search/resume" : "/api/search/start";
        this.postAction(endpoint);
      });
    }
    if (this.btnPause) this.btnPause.addEventListener("click", () => this.postAction("/api/search/pause"));
    if (this.btnReset) {
      this.btnReset.addEventListener("click", () => {
        this.hasManualViewport = false;
        this.postAction("/api/search/reset");
      });
    }

    if (this.speedSlider) {
      let configTimer = null;
      this.speedSlider.addEventListener("input", (e) => {
        const ms = Number(e.target.value);
        if (this.speedValue) this.speedValue.textContent = `${ms} ms`;
        clearTimeout(configTimer);
        configTimer = setTimeout(() => this.updateStepDelay(ms), 120);
      });
    }

    if (this.btnTheme) {
      this.btnTheme.addEventListener("click", () => {
        const current = document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark";
        this.applyTheme(current === "light" ? "dark" : "light");
      });
    }

    if (this.btnFollowBest) {
      this.btnFollowBest.addEventListener("click", () => {
        this.setAutoFollow(!this.autoFollow);
        if (this.autoFollow && this.treeData) {
          const target = this.resolveFollowNode();
          if (target) this.selectNode(target, false);
        }
      });
    }

    if (this.tabOverview && this.tabRaw) {
      this.tabOverview.addEventListener("click", () => this.setInspectorMode("overview"));
      this.tabRaw.addEventListener("click", () => this.setInspectorMode("raw"));
    }

    if (this.thoughtDisplay) {
      this.thoughtDisplay.addEventListener("scroll", () => {
        if (!this.liveStream.active) return;
        const dist =
          this.thoughtDisplay.scrollHeight -
          this.thoughtDisplay.scrollTop -
          this.thoughtDisplay.clientHeight;
        this.liveAutoScroll = dist <= 24;
      });
    }

    // Pan & Zoom Controls
    if (this.btnZoomIn) this.btnZoomIn.addEventListener("click", () => this.zoomBy(1.2));
    if (this.btnZoomOut) this.btnZoomOut.addEventListener("click", () => this.zoomBy(1 / 1.2));
    if (this.btnFitTree) {
      this.btnFitTree.addEventListener("click", () => {
        this.hasManualViewport = false;
        this.fitTreeToView();
      });
    }

    this.bindCanvasPanZoom();
    this.bindKeyboardShortcuts();
  }

  bindCanvasPanZoom() {
    if (!this.treeContainer) return;

    this.treeContainer.addEventListener("mousedown", (e) => {
      if (e.button !== 0 || e.target.closest(".canvas-controls") || e.target.closest(".node")) return;
      this.isPanning = true;
      this.hasManualViewport = true;
      this.treeContainer.classList.add("panning");
      this.panStart = { x: e.clientX - this.panX, y: e.clientY - this.panY };
    });

    window.addEventListener("mousemove", (e) => {
      if (!this.isPanning) return;
      this.panX = e.clientX - this.panStart.x;
      this.panY = e.clientY - this.panStart.y;
      this.applyViewportTransform();
    });

    window.addEventListener("mouseup", () => {
      if (!this.isPanning) return;
      this.isPanning = false;
      this.treeContainer.classList.remove("panning");
    });

    this.treeContainer.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        const rect = this.treeContainer.getBoundingClientRect();
        const cursorX = e.clientX - rect.left;
        const cursorY = e.clientY - rect.top;
        const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
        this.zoomAroundPoint(factor, cursorX, cursorY);
      },
      { passive: false }
    );
  }

  bindKeyboardShortcuts() {
    document.addEventListener("keydown", (e) => {
      const tag = e.target && e.target.tagName ? e.target.tagName.toLowerCase() : "";
      if (tag === "input" || tag === "textarea" || tag === "select") return;

      if (e.code === "Space") {
        e.preventDefault();
        if (this.status === "running") {
          this.postAction("/api/search/pause");
        } else if (this.status !== "finished") {
          const endpoint = this.status === "paused" ? "/api/search/resume" : "/api/search/start";
          this.postAction(endpoint);
        }
      } else if (e.key === "s" || e.key === "S") {
        if (this.status !== "running" && this.status !== "finished") {
          this.postAction("/api/search/step");
        }
      } else if (e.key === "r" || e.key === "R") {
        this.hasManualViewport = false;
        this.postAction("/api/search/reset");
      } else if (e.key === "f" || e.key === "F") {
        this.hasManualViewport = false;
        this.fitTreeToView();
      }
    });
  }

  setAutoFollow(enabled) {
    this.autoFollow = enabled;
    if (this.btnFollowBest) {
      this.btnFollowBest.textContent = `Auto-follow: ${enabled ? "ON" : "OFF"}`;
      this.btnFollowBest.classList.toggle("active", enabled);
      this.btnFollowBest.setAttribute("aria-pressed", String(enabled));
    }
  }

  setInspectorMode(mode) {
    this.inspectorMode = mode;
    const isOverview = mode === "overview";
    if (this.tabOverview) {
      this.tabOverview.classList.toggle("active", isOverview);
      this.tabOverview.setAttribute("aria-selected", String(isOverview));
    }
    if (this.tabRaw) {
      this.tabRaw.classList.toggle("active", !isOverview);
      this.tabRaw.setAttribute("aria-selected", String(!isOverview));
    }
    if (this.metricsOverview) this.metricsOverview.hidden = !isOverview;
    if (this.nodeInspector) this.nodeInspector.hidden = isOverview;
    if (this.stateOverview) this.stateOverview.hidden = !isOverview;
    if (this.stateInspector) this.stateInspector.hidden = isOverview;
  }

  showToast(message) {
    if (!this.toastContainer) return;
    const toast = document.createElement("div");
    toast.className = "toast";
    toast.textContent = message;
    this.toastContainer.appendChild(toast);
    setTimeout(() => toast.remove(), 7000);
  }

  async updateStepDelay(ms) {
    try {
      const res = await fetch("/api/search/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ step_delay_ms: ms }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
    } catch (err) {
      console.error("Failed to update step delay:", err);
      this.showToast("Could not update step delay.");
    }
  }

  async postAction(endpoint) {
    const isStep = endpoint.endsWith("/step");
    if (isStep) {
      if (this.isStepping) return;
      this.isStepping = true;
      this.setStepPending(true);
    }
    try {
      const res = await fetch(endpoint, { method: "POST" });
      if (!res.ok) {
        let detail = `HTTP ${res.status}`;
        try {
          const errJson = await res.json();
          if (errJson && errJson.detail) detail = errJson.detail;
        } catch (_) {}
        throw new Error(detail);
      }
      const data = await res.json();

      if (data.max_iterations !== undefined) {
        this.maxIterations = data.max_iterations;
      }
      if (Array.isArray(data.best_path_ids)) {
        this.bestPathIds = new Set(data.best_path_ids);
      }
      this.updateStatus(data.status, data.iteration);

      if (endpoint.endsWith("/reset")) {
        this.selectedNode = null;
      }
      this.liveStream.active = false;
      this.liveElements = null;
      if (data.tree) {
        this.treeData = data.tree;
        this.syncTreeAndSelection(endpoint.endsWith("/reset"));
      } else {
        await this.fetchTreeAndState();
      }
    } catch (err) {
      console.error(`Error calling ${endpoint}:`, err);
      this.liveStream.active = false;
      this.liveElements = null;
      this.showToast(`Action failed (${endpoint.split("/").pop()}): ${err.message}`);
      await this.fetchTreeAndState();
    } finally {
      if (isStep) {
        this.isStepping = false;
        this.setStepPending(false);
        this.updateStatus(this.status, this.iteration);
      }
    }
  }

  setStepPending(pending) {
    if (this.stepTicker) {
      clearInterval(this.stepTicker);
      this.stepTicker = null;
    }
    if (this.btnStep) {
      this.btnStep.disabled = pending;
      const label = this.btnStep.querySelector("span");
      if (label) {
        if (pending) {
          this.stepStartTime = Date.now();
          label.textContent = "Stepping...";
          this.stepTicker = setInterval(() => {
            const secs = Math.floor((Date.now() - this.stepStartTime) / 1000);
            if (secs >= 1) label.textContent = `Stepping (${secs}s)...`;
          }, 1000);
        } else {
          label.textContent = "Step";
        }
      }
    }
    if (this.btnPlay && pending) {
      this.btnPlay.disabled = true;
    }
    this.updateActiveSyncPolling();
  }

  updateActiveSyncPolling() {
    const needsPoll = this.isStepping || this.status === "running";
    if (needsPoll && !this.syncTimer) {
      this.syncTimer = setInterval(() => this.pollLatestState(), 1200);
    } else if (!needsPoll && this.syncTimer) {
      clearInterval(this.syncTimer);
      this.syncTimer = null;
    }
  }

  async pollLatestState() {
    try {
      const res = await fetch("/api/meta");
      if (!res.ok) return;
      const meta = await res.json();
      if (meta.iteration !== this.iteration || (meta.status && meta.status.toLowerCase() !== this.status)) {
        this.applyMeta(meta);
        await this.fetchTreeAndState();
      }
    } catch (_) {
      // Ignore transient poll errors while request is in flight
    }
  }

  async fetchInitialData() {
    try {
      const [metaRes, treeRes, stateRes] = await Promise.all([
        fetch("/api/meta"),
        fetch("/api/tree"),
        fetch("/api/state"),
      ]);

      if (metaRes.ok) {
        const meta = await metaRes.json();
        this.applyMeta(meta);
      }
      if (treeRes.ok) {
        this.treeData = await treeRes.json();
      }
      if (stateRes.ok) {
        this.stateData = await stateRes.json();
        this.renderTabletop();
      }
      if (this.treeData) {
        this.syncTreeAndSelection(false);
      }
    } catch (err) {
      console.error("Error fetching initial visualizer data:", err);
      this.showToast("Failed to load initial visualizer state.");
    }
  }

  async fetchTreeAndState() {
    try {
      const [treeRes, stateRes] = await Promise.all([
        fetch("/api/tree"),
        fetch("/api/state"),
      ]);
      if (treeRes.ok) this.treeData = await treeRes.json();
      if (stateRes.ok) {
        this.stateData = await stateRes.json();
        this.renderTabletop();
      }
      if (this.treeData) this.syncTreeAndSelection(false);
    } catch (err) {
      console.error("Error refreshing tree/state:", err);
    }
  }

  applyMeta(meta) {
    if (!meta) return;
    if (meta.max_iterations !== undefined) this.maxIterations = meta.max_iterations;
    if (meta.step_delay_ms !== undefined && this.speedSlider) {
      this.speedSlider.value = String(meta.step_delay_ms);
      if (this.speedValue) this.speedValue.textContent = `${meta.step_delay_ms} ms`;
    }
    if (meta.mode && this.modeBadge) {
      this.modeBadge.textContent = meta.mode;
      this.modeBadge.hidden = false;
    }
    if (Array.isArray(meta.best_path_ids)) {
      this.bestPathIds = new Set(meta.best_path_ids);
    }
    this.renderGoal(meta.goal);
    this.updateStatus(meta.status, meta.iteration);
  }

  renderGoal(goal) {
    if (!this.goalStrip || !this.goalText) return;
    if (!goal || (Array.isArray(goal) && goal.length === 0)) {
      this.goalStrip.hidden = true;
      return;
    }
    let formatted = "";
    if (Array.isArray(goal)) {
      formatted = goal.join(" · ");
    } else if (typeof goal === "object") {
      formatted = Object.entries(goal)
        .map(([top, bottom]) => `${top} on ${bottom}`)
        .join(" · ");
    } else {
      formatted = String(goal);
    }
    this.goalText.textContent = formatted;
    this.goalStrip.hidden = false;
  }

  connectSSE() {
    if (this.eventSource) {
      this.eventSource.close();
    }
    this.setSSEState("connecting");
    this.eventSource = new EventSource("/api/stream");

    this.eventSource.onopen = () => {
      this.setSSEState("connected");
    };

    this.eventSource.onmessage = (event) => {
      this.setSSEState("connected");
      try {
        const payload = JSON.parse(event.data);
        this.handleEvent(payload);
      } catch (err) {
        console.error("SSE parse error:", err);
      }
    };

    this.eventSource.onerror = () => {
      this.setSSEState("disconnected");
    };
  }

  setSSEState(state) {
    if (!this.sseDot) return;
    this.sseDot.className = `sse-dot ${state}`;
    const titles = {
      connected: "Live stream connected",
      connecting: "Connecting to stream...",
      disconnected: "Stream disconnected, reconnecting...",
    };
    this.sseDot.title = titles[state] || state;
  }

  handleEvent(payload) {
    if (payload.max_iterations !== undefined) {
      this.maxIterations = payload.max_iterations;
    }
    if (Array.isArray(payload.best_path_ids)) {
      this.bestPathIds = new Set(payload.best_path_ids);
    }
    if (payload.status) {
      this.updateStatus(payload.status, payload.iteration);
    }

    if (payload.type === "error" && payload.message) {
      this.liveStream.active = false;
      this.liveElements = null;
      this.showToast(payload.message);
    }

    if (payload.type === "phase") {
      this.liveStream.active = true;
      if (payload.iteration !== undefined) this.liveStream.iteration = payload.iteration;
      this.liveStream.phase = payload.phase || "";

      if (payload.tree) {
        this.treeData = payload.tree;
        this.renderTree();
        this.renderTrajectoryBar();
      }
      if (payload.preview_state && this.autoFollow) {
        this.stateData = payload.preview_state;
        this.renderTabletop();
        this.renderStateOverview(payload.preview_state, payload.preview_state);
      }

      if (this.autoFollow) {
        this.renderLiveStream();
      }
      return;
    }

    if (payload.type === "token") {
      const incomingCallId = payload.call_id ?? null;
      if (!this.liveStream.active || (incomingCallId !== null && this.liveStream.callId !== incomingCallId)) {
        this.liveStream.active = true;
        this.liveStream.callId = incomingCallId;
        this.liveStream.reasoning = "";
        this.liveStream.content = "";
        this.liveElements = null;
        this.liveAutoScroll = true;
      }
      if (payload.iteration !== undefined) this.liveStream.iteration = payload.iteration;
      if (payload.kind === "reasoning") {
        this.liveStream.reasoning += payload.text || "";
      } else {
        this.liveStream.content += payload.text || "";
      }
      if (this.autoFollow) {
        this.renderLiveStream();
      }
      return;
    }

    const isReset = payload.type === "reset";
    if (isReset || payload.type === "step") {
      this.liveStream.active = false;
      this.liveElements = null;
      if (isReset) {
        this.liveStream.phase = "";
        this.liveStream.callId = null;
        this.liveStream.reasoning = "";
        this.liveStream.content = "";
        this.liveAutoScroll = true;
      }
    }
    if (isReset) {
      this.selectedNode = null;
      this.hasManualViewport = false;
    }

    if (payload.tree) {
      this.treeData = payload.tree;
      this.syncTreeAndSelection(isReset);
    }
  }

  syncTreeAndSelection(isReset) {
    if (!this.treeData) return;

    // Ensure bestPathIds has a valid fallback when backend controller has no mcts attached
    if (this.bestPathIds.size === 0) {
      const fallbackPath = this.computeBestPathFallback(this.treeData);
      this.bestPathIds = new Set(fallbackPath.map((n) => n.id));
    }

    this.renderTree();
    this.renderTrajectoryBar();

    if (isReset) {
      this.selectNode(this.treeData, false);
      return;
    }

    if (this.autoFollow) {
      const target = this.resolveFollowNode();
      if (target) this.selectNode(target, false);
    } else if (this.selectedNode) {
      const refreshed = this.findNodeById(this.treeData, this.selectedNode.id);
      this.selectNode(refreshed || this.treeData, false);
    } else {
      this.selectNode(this.treeData, false);
    }
  }

  updateStatus(status, iteration) {
    this.status = (status || this.status || "idle").toLowerCase();
    this.iteration = iteration !== undefined ? iteration : this.iteration;

    if (this.statusText) this.statusText.textContent = this.status.toUpperCase();
    if (this.iterationCounter) {
      this.iterationCounter.textContent = `${this.iteration} / ${this.maxIterations}`;
    }
    if (this.progressFill) {
      const pct = this.maxIterations > 0
        ? Math.min(100, Math.round((this.iteration / this.maxIterations) * 100))
        : 0;
      this.progressFill.style.width = `${pct}%`;
    }
    if (this.statusDot) this.statusDot.className = `status-dot ${this.status}`;

    const isRunning = this.status === "running";
    const isFinished = this.status === "finished" || this.iteration >= this.maxIterations;

    if (this.btnStep) this.btnStep.disabled = isRunning || isFinished;
    if (this.btnPlay) this.btnPlay.disabled = isRunning || isFinished;
    if (this.btnPause) this.btnPause.disabled = !isRunning;
    if (this.btnPlayLabel) {
      this.btnPlayLabel.textContent = this.status === "paused" ? "Resume" : "Play";
    }
    this.updateActiveSyncPolling();
  }

  computeBestPathFallback(root) {
    const path = [];
    let curr = root;
    while (curr) {
      path.push(curr);
      if (!curr.children || curr.children.length === 0) break;
      const visited = curr.children.filter((c) => (c.visits || 0) > 0);
      if (visited.length === 0) break;
      curr = visited.reduce((best, c) => {
        if ((c.visits || 0) !== (best.visits || 0)) {
          return (c.visits || 0) > (best.visits || 0) ? c : best;
        }
        return (c.Q || 0) > (best.Q || 0) ? c : best;
      }, visited[0]);
    }
    return path;
  }

  collectBestPathNodes(root) {
    const path = [];
    let curr = root;
    while (curr && this.bestPathIds.has(curr.id)) {
      path.push(curr);
      if (!curr.children || curr.children.length === 0) break;
      curr = curr.children.find((c) => this.bestPathIds.has(c.id)) || null;
    }
    return path.length > 0 ? path : this.computeBestPathFallback(root);
  }

  resolveFollowNode() {
    if (!this.treeData) return null;
    const path = this.collectBestPathNodes(this.treeData);
    return path[path.length - 1] || this.treeData;
  }

  findNodeById(root, id) {
    if (!root) return null;
    if (root.id === id) return root;
    if (root.children) {
      for (const child of root.children) {
        const res = this.findNodeById(child, id);
        if (res) return res;
      }
    }
    return null;
  }

  /* ========================================================================
     SVG Tree Rendering + Pan & Zoom
     ======================================================================== */

  renderTree() {
    if (!this.treeData || !this.treeViewport) return;

    this.treeViewport.innerHTML = "";
    this.nodeElements.clear();

    const nodes = [];
    const links = [];
    let nextX = 72;

    const assignCoords = (node, depth) => {
      if (!node.children || node.children.length === 0) {
        node.x = nextX;
        nextX += 132;
      } else {
        node.children.forEach((c) => assignCoords(c, depth + 1));
        const first = node.children[0].x;
        const last = node.children[node.children.length - 1].x;
        node.x = (first + last) / 2;
      }
      node.y = 56 + depth * 92;
      nodes.push(node);

      if (node.children) {
        node.children.forEach((c) => links.push({ source: node, target: c }));
      }
    };

    assignCoords(this.treeData, 0);

    const maxDepth = nodes.reduce((m, n) => Math.max(m, n.depth || 0), 0);
    this.treeBounds = {
      width: Math.max(nextX + 40, 220),
      height: Math.max(120 + maxDepth * 92, 220),
    };

    if (this.treeSummary) {
      const plural = nodes.length === 1 ? "node" : "nodes";
      this.treeSummary.textContent = `${nodes.length} ${plural} · depth ${maxDepth}`;
    }

    // Draw Links first (non-best first, best-path on top)
    const sortedLinks = [...links].sort((a, b) => {
      const aBest = this.bestPathIds.has(a.source.id) && this.bestPathIds.has(a.target.id) ? 1 : 0;
      const bBest = this.bestPathIds.has(b.source.id) && this.bestPathIds.has(b.target.id) ? 1 : 0;
      return aBest - bBest;
    });

    sortedLinks.forEach((l) => {
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      const midY = (l.source.y + l.target.y) / 2;
      const d = `M ${l.source.x} ${l.source.y} C ${l.source.x} ${midY}, ${l.target.x} ${midY}, ${l.target.x} ${l.target.y}`;
      path.setAttribute("d", d);

      const isBest = this.bestPathIds.has(l.source.id) && this.bestPathIds.has(l.target.id);
      const isVisited = (l.target.visits || 0) > 0;
      const isPreview = Boolean(l.target.is_preview);
      path.setAttribute(
        "class",
        `link ${isPreview ? "preview" : isBest ? "best-path" : isVisited ? "visited" : ""}`
      );
      this.treeViewport.appendChild(path);
    });

    // Draw Nodes
    nodes.forEach((n) => {
      const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
      const isSelected = this.selectedNode && this.selectedNode.id === n.id;
      const isBest = this.bestPathIds.has(n.id);
      const isVisited = (n.visits || 0) > 0;
      const isPreview = Boolean(n.is_preview);

      const classes = ["node"];
      if (isPreview) classes.push("preview");
      if (isVisited) classes.push("visited");
      if (isBest) classes.push("best-path");
      if (n.is_terminal) classes.push("terminal");
      if (isSelected) classes.push("active");

      g.setAttribute("class", classes.join(" "));
      g.setAttribute("transform", `translate(${n.x}, ${n.y})`);

      const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      circle.setAttribute("r", 20);
      circle.addEventListener("click", (e) => {
        e.stopPropagation();
        this.hideTooltip();
        this.selectNode(n, true);
      });
      circle.addEventListener("mouseenter", (e) => this.showTooltip(n, e));
      circle.addEventListener("mousemove", (e) => this.moveTooltip(e));
      circle.addEventListener("mouseleave", () => this.hideTooltip());
      g.appendChild(circle);

      const visitsText = document.createElementNS("http://www.w3.org/2000/svg", "text");
      visitsText.setAttribute("class", "node-visits");
      visitsText.setAttribute("dy", 4);
      visitsText.textContent = `N:${n.visits}`;
      g.appendChild(visitsText);

      const actionLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
      actionLabel.setAttribute("class", "node-action");
      actionLabel.setAttribute("dy", 33);
      actionLabel.textContent = this.formatShortAction(n);
      g.appendChild(actionLabel);

      if (n.visits > 0) {
        const qLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
        qLabel.setAttribute("class", "node-q");
        qLabel.setAttribute("dy", 45);
        qLabel.textContent = `Q:${Number(n.Q || 0).toFixed(1)}`;
        g.appendChild(qLabel);
      }

      this.nodeElements.set(n.id, g);
      this.treeViewport.appendChild(g);
    });

    if (!this.hasManualViewport) {
      this.fitTreeToView();
    } else {
      this.applyViewportTransform();
    }
  }

  formatShortAction(node) {
    if (!node.action) return node.depth === 0 ? "Root" : "Step";
    const s = String(node.action);
    const cleaned = s
      .replace(/^BlockAction\(/i, "")
      .replace(/Pick up the ([a-zA-Z0-9_-]+) block\.?/i, "pick $1")
      .replace(/Put down the ([a-zA-Z0-9_-]+) block\.?/i, "drop $1")
      .replace(/Stack the ([a-zA-Z0-9_-]+) block on top of the ([a-zA-Z0-9_-]+) block\.?/i, "stack $1→$2")
      .replace(/Unstack the ([a-zA-Z0-9_-]+) block from on top of the ([a-zA-Z0-9_-]+) block\.?/i, "unstack $1←$2");
    return cleaned.length > 18 ? `${cleaned.slice(0, 17)}…` : cleaned;
  }

  fitTreeToView() {
    if (!this.treeContainer) return;
    const rect = this.treeContainer.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;

    const padding = 36;
    const scaleX = (rect.width - padding) / this.treeBounds.width;
    const scaleY = (rect.height - padding) / this.treeBounds.height;
    this.zoom = Math.max(0.35, Math.min(1.15, Math.min(scaleX, scaleY)));
    this.panX = (rect.width - this.treeBounds.width * this.zoom) / 2;
    this.panY = Math.max(12, (rect.height - this.treeBounds.height * this.zoom) / 2);
    this.applyViewportTransform();
  }

  zoomBy(factor) {
    if (!this.treeContainer) return;
    const rect = this.treeContainer.getBoundingClientRect();
    this.zoomAroundPoint(factor, rect.width / 2, rect.height / 2);
  }

  zoomAroundPoint(factor, cx, cy) {
    const nextZoom = Math.max(0.25, Math.min(2.5, this.zoom * factor));
    const ratio = nextZoom / this.zoom;
    this.panX = cx - (cx - this.panX) * ratio;
    this.panY = cy - (cy - this.panY) * ratio;
    this.zoom = nextZoom;
    this.hasManualViewport = true;
    this.applyViewportTransform();
  }

  applyViewportTransform() {
    if (this.treeViewport) {
      this.treeViewport.setAttribute("transform", `translate(${this.panX}, ${this.panY}) scale(${this.zoom})`);
    }
    if (this.zoomReadout) {
      this.zoomReadout.textContent = `${Math.round(this.zoom * 100)}%`;
    }
  }

  showTooltip(node, event) {
    if (!this.treeTooltip) return;
    const actionText = node.action ? String(node.action) : "Root State";
    const mStr = node.M === null || node.M === undefined ? "—" : Number(node.M).toFixed(2);
    const uctStr = node.uct === null || node.uct === undefined ? "—" : Number(node.uct).toFixed(2);
    this.treeTooltip.innerHTML = `
      <div class="tt-action">${this.escapeHtml(actionText)}</div>
      <div class="tt-metrics">
        <span>N=${node.visits}</span>
        <span>Q=${Number(node.Q || 0).toFixed(2)}</span>
        <span>M=${mStr}</span>
        <span>UCT=${uctStr}</span>
        <span>R=${Number(node.reward || 0).toFixed(1)}</span>
      </div>
    `;
    this.treeTooltip.hidden = false;
    this.moveTooltip(event);
  }

  moveTooltip(event) {
    if (!this.treeTooltip || this.treeTooltip.hidden || !this.treeContainer) return;
    const rect = this.treeContainer.getBoundingClientRect();
    const x = Math.min(rect.width - 200, Math.max(8, event.clientX - rect.left + 12));
    const y = Math.min(rect.height - 70, Math.max(8, event.clientY - rect.top + 12));
    this.treeTooltip.style.left = `${x}px`;
    this.treeTooltip.style.top = `${y}px`;
  }

  hideTooltip() {
    if (this.treeTooltip) this.treeTooltip.hidden = true;
  }

  /* ========================================================================
     Trajectory Breadcrumb & Node Inspector
     ======================================================================== */

  renderTrajectoryBar() {
    if (!this.trajectoryStrip || !this.treeData) return;
    this.bestPathNodes = this.collectBestPathNodes(this.treeData);
    this.trajectoryStrip.innerHTML = "";

    this.bestPathNodes.forEach((node, idx) => {
      if (idx > 0) {
        const sep = document.createElement("span");
        sep.className = "traj-arrow";
        sep.textContent = "›";
        this.trajectoryStrip.appendChild(sep);
      }
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "traj-step";
      if (this.selectedNode && this.selectedNode.id === node.id) {
        btn.classList.add("active");
      }
      btn.textContent = idx === 0 ? "Root" : this.formatShortAction(node);
      btn.title = node.action ? String(node.action) : "Initial state";
      btn.addEventListener("click", () => this.selectNode(node, true));
      this.trajectoryStrip.appendChild(btn);
    });
  }

  selectNode(node, isUserClick = false) {
    if (!node) return;

    if (isUserClick) {
      this.setAutoFollow(false);
      this.liveStream.active = false;
      this.liveElements = null;
    }

    this.selectedNode = node;

    // Immediately sync SVG active ring & trajectory active pill
    this.nodeElements.forEach((el, id) => {
      el.classList.toggle("active", id === node.id);
    });
    if (this.trajectoryStrip) {
      const pills = this.trajectoryStrip.querySelectorAll(".traj-step");
      this.bestPathNodes.forEach((pNode, i) => {
        if (pills[i]) pills[i].classList.toggle("active", pNode.id === node.id);
      });
    }

    this.renderInspector(node);
    this.updateTabletopForNode(node);
  }

  renderLiveStream() {
    if (!this.thoughtDisplay || !this.liveStream.active) return;

    const phaseLabel = this.liveStream.phase ? this.liveStream.phase.toUpperCase() : "LLM";
    const iterLabel = this.liveStream.iteration ? `Iter ${this.liveStream.iteration} · ` : "";
    const callLabel = this.liveStream.callId ? ` · Call #${this.liveStream.callId}` : "";

    if (this.thoughtModeTag) {
      this.thoughtModeTag.textContent = `LIVE · ${iterLabel}${phaseLabel}${callLabel}`;
      this.thoughtModeTag.classList.add("live");
    }

    if (!this.liveElements) {
      this.thoughtDisplay.innerHTML = "";
      const waitingEl = document.createElement("p");
      waitingEl.className = "empty-hint";

      const reasoningLabel = document.createElement("div");
      reasoningLabel.className = "live-section-label";
      reasoningLabel.textContent = "Thinking (Live)";
      reasoningLabel.hidden = true;

      const reasoningPre = document.createElement("pre");
      reasoningPre.className = "live-stream-pre live-reasoning";
      reasoningPre.hidden = true;

      const outputLabel = document.createElement("div");
      outputLabel.className = "live-section-label";
      outputLabel.textContent = "Output (Live)";
      outputLabel.hidden = true;

      const outputPre = document.createElement("pre");
      outputPre.className = "live-stream-pre live-output";
      outputPre.hidden = true;

      this.thoughtDisplay.appendChild(waitingEl);
      this.thoughtDisplay.appendChild(reasoningLabel);
      this.thoughtDisplay.appendChild(reasoningPre);
      this.thoughtDisplay.appendChild(outputLabel);
      this.thoughtDisplay.appendChild(outputPre);

      this.liveElements = {
        waitingEl,
        reasoningLabel,
        reasoningPre,
        outputLabel,
        outputPre,
      };
    }

    const els = this.liveElements;
    const hasReasoning = Boolean(this.liveStream.reasoning);
    const hasContent = Boolean(this.liveStream.content);

    els.waitingEl.hidden = hasReasoning || hasContent;
    if (els.waitingEl.hidden === false) {
      els.waitingEl.innerHTML = `MCTS phase <strong>${this.escapeHtml(phaseLabel)}</strong> — waiting for model stream...`;
    }

    els.reasoningLabel.hidden = !hasReasoning;
    els.reasoningPre.hidden = !hasReasoning;
    if (hasReasoning) {
      els.reasoningPre.textContent = this.liveStream.reasoning;
    }

    els.outputLabel.hidden = !hasContent;
    els.outputPre.hidden = !hasContent;
    if (hasContent) {
      els.outputPre.textContent = this.liveStream.content;
    }

    if (this.liveAutoScroll) {
      this.thoughtDisplay.scrollTop = this.thoughtDisplay.scrollHeight;
    }
  }

  renderInspector(node) {
    if (this.nodeIdBadge) this.nodeIdBadge.textContent = node.id || "—";
    if (this.terminalBadge) this.terminalBadge.hidden = !node.is_terminal;

    // 1. Thought & Action Card
    if (this.thoughtModeTag) {
      this.thoughtModeTag.classList.remove("live");
      if (node.thought) this.thoughtModeTag.textContent = "Hybrid Thought";
      else if (node.action) this.thoughtModeTag.textContent = "Paper Action";
      else this.thoughtModeTag.textContent = "Root";
    }

    if (this.thoughtDisplay) {
      if (node.thought) {
        this.thoughtDisplay.innerHTML = `
          <div class="thought-quote">${this.escapeHtml(node.thought)}</div>
          ${node.action ? `<div class="action-line"><span>Action</span><span class="action-pill">${this.escapeHtml(node.action)}</span></div>` : ""}
        `;
      } else if (node.action) {
        this.thoughtDisplay.innerHTML = `
          <div class="action-line"><span>Action</span><span class="action-pill">${this.escapeHtml(node.action)}</span></div>
          <p class="empty-hint">Natural-language Paper mode transition (no structured thought recorded).</p>
        `;
      } else {
        this.thoughtDisplay.innerHTML = `
          <p class="empty-hint">Initial starting state. Use <strong>Step</strong> or <strong>Play</strong> to expand the reasoning tree.</p>
        `;
      }
    }

    // 2. Metrics Overview + Raw JSON
    if (this.metricsOverview) {
      const fmt = (v, digits = 2) => (v === null || v === undefined ? "—" : Number(v).toFixed(digits));
      const items = [
        ["Visits", String(node.visits ?? 0)],
        ["Mean Q", fmt(node.Q)],
        ["Max M", fmt(node.M)],
        ["Reward", fmt(node.reward, 1)],
        ["UCT", fmt(node.uct)],
        ["Depth", String(node.depth ?? 0)],
        ["Terminal", node.is_terminal ? "Yes" : "No"],
        ["Children", String((node.children || []).length)],
      ];
      this.metricsOverview.innerHTML = items
        .map(
          ([label, val]) => `
            <div class="kpi-cell">
              <span class="kpi-label">${label}</span>
              <span class="kpi-val">${this.escapeHtml(val)}</span>
            </div>
          `
        )
        .join("");
    }

    if (this.nodeInspector) {
      this.nodeInspector.textContent = JSON.stringify(
        {
          id: node.id,
          action: node.action,
          visits: node.visits,
          Q: node.Q,
          M: node.M,
          reward: node.reward,
          uct: node.uct,
          is_terminal: node.is_terminal,
          depth: node.depth,
        },
        null,
        2
      );
    }

    // 3. Physical State Overview + Raw JSON
    if (this.stateInspector) {
      this.stateInspector.textContent =
        typeof node.state === "object" && node.state !== null
          ? JSON.stringify(node.state, null, 2)
          : String(node.state ?? "");
    }
  }

  renderStateOverview(parsedState, rawState) {
    if (!this.stateOverview) return;
    if (parsedState && (Array.isArray(parsedState.stacks) || parsedState.holding !== undefined)) {
      const stacksText =
        parsedState.stacks && parsedState.stacks.length > 0
          ? parsedState.stacks.map((s) => `[${s.join(" → ")}]`).join("  ")
          : "None";
      const clearText =
        parsedState.clear && parsedState.clear.length > 0 ? parsedState.clear.join(", ") : "—";
      const holdingText = parsedState.holding || "Empty";

      this.stateOverview.innerHTML = `
        <div class="state-row"><span class="state-key">Holding</span><span class="state-val">${this.escapeHtml(holdingText)}</span></div>
        <div class="state-row"><span class="state-key">Stacks</span><span class="state-val">${this.escapeHtml(stacksText)}</span></div>
        <div class="state-row"><span class="state-key">Clear</span><span class="state-val">${this.escapeHtml(clearText)}</span></div>
      `;
    } else {
      this.stateOverview.innerHTML = `
        <div class="state-row"><span class="state-key">State</span><span class="state-val">${this.escapeHtml(String(rawState ?? "—"))}</span></div>
      `;
    }
  }

  /* ========================================================================
     Blocksworld Tabletop Simulator
     ======================================================================== */

  parseTextState(text) {
    if (!text || typeof text !== "string") return null;
    const holdingMatch = text.match(/(?:hand\s+is\s+holding\s+the|holding\s+the)\s+([a-zA-Z0-9_-]+)\s+block/i);
    const holding = holdingMatch ? holdingMatch[1].toLowerCase() : null;

    const onMatches = [
      ...text.matchAll(/the\s+([a-zA-Z0-9_-]+)\s+block\s+is\s+on\s+top\s+of\s+the\s+([a-zA-Z0-9_-]+)\s+block/gi),
    ];
    const onMap = {};
    onMatches.forEach((m) => {
      onMap[m[2].toLowerCase()] = m[1].toLowerCase();
    });

    const tableMatches = [
      ...text.matchAll(/the\s+([a-zA-Z0-9_-]+)\s+block\s+is\s+on\s+the\s+table/gi),
    ];
    const tableBlocks = tableMatches.map((m) => m[1].toLowerCase()).sort();

    const clearMatches = [...text.matchAll(/the\s+([a-zA-Z0-9_-]+)\s+block\s+is\s+clear/gi)];
    const clear = clearMatches.map((m) => m[1].toLowerCase()).sort();

    if (!holding && tableBlocks.length === 0 && onMatches.length === 0) {
      return null;
    }

    const stacks = [];
    tableBlocks.forEach((bottom) => {
      const stack = [bottom];
      let curr = bottom;
      while (onMap[curr]) {
        curr = onMap[curr];
        stack.push(curr);
      }
      stacks.push(stack);
    });

    return { stacks, holding, clear };
  }

  updateTabletopForNode(node) {
    let parsed = null;
    if (node && node.state) {
      if (typeof node.state === "object" && Array.isArray(node.state.stacks)) {
        parsed = node.state;
        this.stateData = node.state;
      } else if (typeof node.state === "string") {
        parsed = this.parseTextState(node.state);
        if (parsed) this.stateData = parsed;
      }
    }
    this.renderStateOverview(parsed || this.stateData, node ? node.state : "");
    this.renderTabletop();
  }

  renderTabletop() {
    if (!this.stacksArea || !this.heldBlockSlot) return;
    this.stacksArea.innerHTML = "";
    this.heldBlockSlot.innerHTML = "";

    const data = this.stateData || {};
    const holding = data.holding || null;
    const stacks = Array.isArray(data.stacks) ? data.stacks : [];

    if (this.holdingPill) {
      this.holdingPill.textContent = holding ? `Holding: ${holding}` : "Hand: empty";
    }

    if (holding) {
      const blockEl = document.createElement("div");
      blockEl.className = `block ${holding.toLowerCase()}`;
      blockEl.textContent = holding;
      this.heldBlockSlot.appendChild(blockEl);
    }

    if (stacks.length === 0 && !holding) {
      const hint = document.createElement("div");
      hint.className = "empty-table-hint";
      hint.textContent = "No block configuration in current state.";
      this.stacksArea.appendChild(hint);
      return;
    }

    stacks.forEach((stack) => {
      const stackEl = document.createElement("div");
      stackEl.className = "block-stack";
      stack.forEach((blockName) => {
        const blockEl = document.createElement("div");
        blockEl.className = `block ${blockName.toLowerCase()}`;
        blockEl.textContent = blockName;
        stackEl.appendChild(blockEl);
      });
      this.stacksArea.appendChild(stackEl);
    });
  }

  escapeHtml(str) {
    return String(str).replace(/[&<>'"]/g, (tag) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      "'": "&#39;",
      '"': "&quot;",
    }[tag] || tag));
  }
}

document.addEventListener("DOMContentLoaded", () => {
  window.app = new VisualizerApp();
});
