/**
 * RAP Real-Time Visualizer Client
 * Handles SSE streaming, interactive SVG tree rendering, Blocksworld tabletop, controls, and theme switching.
 */

class VisualizerApp {
  constructor() {
    this.treeData = null;
    this.stateData = null;
    this.status = "idle";
    this.iteration = 0;
    this.selectedNode = null;
    this.autoFollow = true;
    this.eventSource = null;

    this.initElements();
    this.bindEvents();
    this.connectSSE();
    this.fetchInitialData();
  }

  initElements() {
    this.statusDot = document.getElementById("status-dot");
    this.statusText = document.getElementById("status-text");
    this.iterationCounter = document.getElementById("iteration-counter");
    this.btnStep = document.getElementById("btn-step");
    this.btnPlay = document.getElementById("btn-play");
    this.btnPause = document.getElementById("btn-pause");
    this.btnReset = document.getElementById("btn-reset");
    this.btnFollowBest = document.getElementById("btn-follow-best");
    this.themeSelect = document.getElementById("theme-select");
    this.treeSvg = document.getElementById("tree-svg");
    this.stacksArea = document.getElementById("stacks-area");
    this.heldBlockSlot = document.getElementById("held-block-slot");
    this.thoughtDisplay = document.getElementById("thought-display");
    this.nodeInspector = document.getElementById("node-inspector");
    this.stateInspector = document.getElementById("state-inspector");
  }

  bindEvents() {
    if (this.btnStep) this.btnStep.addEventListener("click", () => this.postAction("/api/search/step"));
    if (this.btnPlay) this.btnPlay.addEventListener("click", () => this.postAction("/api/search/start"));
    if (this.btnPause) this.btnPause.addEventListener("click", () => this.postAction("/api/search/pause"));
    if (this.btnReset) this.btnReset.addEventListener("click", () => this.postAction("/api/search/reset"));
    if (this.btnFollowBest) {
      this.btnFollowBest.addEventListener("click", () => {
        this.autoFollow = !this.autoFollow;
        this.btnFollowBest.textContent = `Auto-Follow Best Path: ${this.autoFollow ? "ON" : "OFF"}`;
        if (this.autoFollow && this.treeData) {
          const best = this.findBestNode(this.treeData);
          if (best) this.selectNode(best, false);
        }
      });
    }

    if (this.themeSelect) {
      const savedTheme = localStorage.getItem("rap-theme") || "tuscan-carbon";
      this.themeSelect.value = savedTheme;
      this.applyTheme(savedTheme);

      this.themeSelect.addEventListener("change", (e) => {
        const theme = e.target.value;
        this.applyTheme(theme);
        localStorage.setItem("rap-theme", theme);
      });
    }
  }

  applyTheme(theme) {
    if (theme === "tuscan-carbon") {
      document.documentElement.removeAttribute("data-theme");
    } else {
      document.documentElement.setAttribute("data-theme", theme);
    }
  }

  async postAction(endpoint) {
    try {
      const res = await fetch(endpoint, { method: "POST" });
      const data = await res.json();
      this.updateStatus(data.status, data.iteration);
      this.fetchInitialData();
    } catch (err) {
      console.error(`Error calling ${endpoint}:`, err);
    }
  }

  async fetchInitialData() {
    try {
      const [treeRes, stateRes] = await Promise.all([
        fetch("/api/tree"),
        fetch("/api/state")
      ]);
      if (treeRes.ok) {
        this.treeData = await treeRes.json();
        this.renderTree();
      }
      if (stateRes.ok) {
        this.stateData = await stateRes.json();
        this.renderTabletop();
      }
    } catch (err) {
      console.error("Error fetching initial visualizer data:", err);
    }
  }

  connectSSE() {
    if (this.eventSource) {
      this.eventSource.close();
    }

    this.eventSource = new EventSource("/api/stream");

    this.eventSource.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        this.handleEvent(payload);
      } catch (err) {
        console.error("SSE parse error:", err);
      }
    };

    this.eventSource.onerror = (err) => {
      console.warn("SSE connection interrupted, retrying...", err);
    };
  }

  handleEvent(payload) {
    if (payload.status) {
      this.updateStatus(payload.status, payload.iteration);
    }
    if (payload.tree) {
      this.treeData = payload.tree;
      this.renderTree();
    }

    // Auto-follow best trajectory node during search
    if (this.autoFollow && this.treeData) {
      const best = this.findBestNode(this.treeData);
      if (best) {
        this.selectNode(best, false);
      }
    }
  }

  updateStatus(status, iteration) {
    this.status = status || this.status;
    this.iteration = iteration !== undefined ? iteration : this.iteration;

    if (this.statusText) this.statusText.textContent = this.status.toUpperCase();
    if (this.iterationCounter) this.iterationCounter.textContent = `Iteration: ${this.iteration}`;
    if (this.statusDot) this.statusDot.className = `status-dot ${this.status.toLowerCase()}`;
    if (this.btnPlay) this.btnPlay.disabled = this.status === "running";
    if (this.btnPause) this.btnPause.disabled = this.status !== "running";
  }

  findBestNode(root) {
    let best = root;
    const traverse = (node) => {
      if (!node) return;
      if (node.visits > best.visits || (node.visits === best.visits && (node.Q || 0) > (best.Q || 0))) {
        best = node;
      }
      if (node.children) {
        node.children.forEach(traverse);
      }
    };
    traverse(root);
    return best;
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

  renderTree() {
    if (!this.treeData || !this.treeSvg) return;

    this.treeSvg.innerHTML = "";

    const nodes = [];
    const links = [];
    let nextX = 40;

    const assignCoords = (node, depth) => {
      if (!node.children || node.children.length === 0) {
        node.x = nextX;
        nextX += 120;
      } else {
        node.children.forEach(c => assignCoords(c, depth + 1));
        const first = node.children[0].x;
        const last = node.children[node.children.length - 1].x;
        node.x = (first + last) / 2;
      }
      node.y = 50 + depth * 80;
      nodes.push(node);

      if (node.children) {
        node.children.forEach(c => {
          links.push({ source: node, target: c });
        });
      }
    };

    assignCoords(this.treeData, 0);

    const totalWidth = Math.max(nextX + 60, 800);
    const maxDepth = Math.max(...nodes.map(n => n.depth || 0));
    const totalHeight = Math.max(100 + maxDepth * 80, 500);

    this.treeSvg.setAttribute("viewBox", `0 0 ${totalWidth} ${totalHeight}`);

    // Draw Links
    links.forEach(l => {
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      const d = `M ${l.source.x} ${l.source.y} C ${l.source.x} ${(l.source.y + l.target.y) / 2}, ${l.target.x} ${(l.source.y + l.target.y) / 2}, ${l.target.x} ${l.target.y}`;
      path.setAttribute("d", d);
      path.setAttribute("class", `link ${l.target.visits > 0 ? "active" : ""}`);
      this.treeSvg.appendChild(path);
    });

    // Draw Nodes
    nodes.forEach(n => {
      const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
      const isSelected = this.selectedNode && this.selectedNode.id === n.id;
      g.setAttribute("class", `node ${n.is_terminal ? "terminal" : ""} ${isSelected ? "active" : ""}`);
      g.setAttribute("transform", `translate(${n.x}, ${n.y})`);

      const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      circle.setAttribute("r", 20);
      circle.addEventListener("click", () => this.selectNode(n, true));
      g.appendChild(circle);

      const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
      text.setAttribute("dy", 4);
      text.textContent = `N:${n.visits}`;
      g.appendChild(text);

      const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
      label.setAttribute("dy", 32);
      label.setAttribute("font-size", "10px");
      label.setAttribute("fill", "var(--text-muted)");
      const actStr = n.action ? String(n.action).slice(0, 15) : (n.depth === 0 ? "Root" : "");
      label.textContent = actStr;
      g.appendChild(label);

      this.treeSvg.appendChild(g);
    });

    if (!this.selectedNode && nodes.length > 0) {
      this.selectNode(nodes[0], false);
    }
  }

  parseTextState(text) {
    if (!text || typeof text !== "string") return null;
    const holdingMatch = text.match(/holding\s+the\s+([a-zA-Z]+)\s+block/i);
    const holding = holdingMatch ? holdingMatch[1].toLowerCase() : null;

    const onMatches = [...text.matchAll(/the\s+([a-zA-Z]+)\s+block\s+is\s+on\s+top\s+of\s+the\s+([a-zA-Z]+)\s+block/gi)];
    const onMap = {};
    onMatches.forEach(m => {
      onMap[m[2].toLowerCase()] = m[1].toLowerCase();
    });

    const tableMatches = [...text.matchAll(/the\s+([a-zA-Z]+)\s+block\s+is\s+on\s+the\s+table/gi)];
    const tableBlocks = tableMatches.map(m => m[1].toLowerCase());

    const stacks = [];
    tableBlocks.forEach(bottom => {
      const stack = [bottom];
      let curr = bottom;
      while (onMap[curr]) {
        curr = onMap[curr];
        stack.push(curr);
      }
      stacks.push(stack);
    });

    return { stacks, holding };
  }

  selectNode(node, isUserClick = false) {
    if (isUserClick) {
      this.autoFollow = false;
      if (this.btnFollowBest) {
        this.btnFollowBest.textContent = "Auto-Follow Best Path: OFF";
      }
    }

    this.selectedNode = node;

    // 1. Update Inspector JSON
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

    if (this.stateInspector) {
      this.stateInspector.textContent = typeof node.state === "object"
        ? JSON.stringify(node.state, null, 2)
        : String(node.state);
    }

    // 2. Update Thought & Reasoning Box
    if (this.thoughtDisplay) {
      const escape = (str) => String(str).replace(/[&<>'"]/g, tag => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"
      }[tag] || tag));

      let content = "";
      if (node.thought) {
        content = `
          <div class="thought-bubble">
            <span class="thought-badge">💡 Model Reasoning (Hybrid Mode)</span>
            <div class="thought-text">"${escape(node.thought)}"</div>
            ${node.action ? `<div class="action-badge">⚡ Action: ${escape(node.action)}</div>` : ""}
          </div>
        `;
      } else if (node.action) {
        content = `
          <div class="thought-bubble">
            <span class="thought-badge">⚡ Action Chosen</span>
            <div class="thought-text">${escape(node.action)}</div>
            <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 4px;">Paper Mode (Pure Natural Language, no structured thought)</div>
          </div>
        `;
      } else {
        content = `
          <div class="thought-bubble">
            <span class="thought-badge">🌱 Root State</span>
            <div class="thought-text">Initial starting configuration. Click Step or Play to begin deliberate planning.</div>
          </div>
        `;
      }
      this.thoughtDisplay.innerHTML = content;
    }

    // 3. Update Tabletop State
    if (node.state) {
      if (typeof node.state === "object" && node.state.stacks) {
        this.stateData = node.state;
      } else if (typeof node.state === "string") {
        const parsed = this.parseTextState(node.state);
        if (parsed) this.stateData = parsed;
      }
      this.renderTabletop();
    }
  }

  renderTabletop() {
    if (!this.stacksArea || !this.heldBlockSlot) return;
    this.stacksArea.innerHTML = "";
    this.heldBlockSlot.innerHTML = "";

    const data = this.stateData || {};
    const holding = data.holding;
    const stacks = data.stacks || [];

    if (holding) {
      const blockEl = document.createElement("div");
      blockEl.className = `block ${holding.toLowerCase()}`;
      blockEl.textContent = holding;
      this.heldBlockSlot.appendChild(blockEl);
    }

    stacks.forEach(stack => {
      const stackEl = document.createElement("div");
      stackEl.className = "block-stack";
      stack.forEach(blockName => {
        const blockEl = document.createElement("div");
        blockEl.className = `block ${blockName.toLowerCase()}`;
        blockEl.textContent = blockName;
        stackEl.appendChild(blockEl);
      });
      this.stacksArea.appendChild(stackEl);
    });
  }
}

document.addEventListener("DOMContentLoaded", () => {
  window.app = new VisualizerApp();
});
