/**
 * RAP Real-Time Visualizer Client
 * Handles SSE streaming, interactive SVG tree rendering, Blocksworld tabletop, and controls.
 */

class VisualizerApp {
  constructor() {
    this.treeData = null;
    this.stateData = null;
    this.status = "idle";
    this.iteration = 0;
    this.selectedNode = null;
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
    this.treeSvg = document.getElementById("tree-svg");
    this.stacksArea = document.getElementById("stacks-area");
    this.heldBlockSlot = document.getElementById("held-block-slot");
    this.nodeInspector = document.getElementById("node-inspector");
    this.stateInspector = document.getElementById("state-inspector");
  }

  bindEvents() {
    this.btnStep.addEventListener("click", () => this.postAction("/api/search/step"));
    this.btnPlay.addEventListener("click", () => this.postAction("/api/search/start"));
    this.btnPause.addEventListener("click", () => this.postAction("/api/search/pause"));
    this.btnReset.addEventListener("click", () => this.postAction("/api/search/reset"));
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
    } else {
      this.fetchInitialData();
    }
  }

  updateStatus(status, iteration) {
    this.status = status || this.status;
    this.iteration = iteration !== undefined ? iteration : this.iteration;

    this.statusText.textContent = this.status.toUpperCase();
    this.iterationCounter.textContent = `Iteration: ${this.iteration}`;

    this.statusDot.className = `status-dot ${this.status.toLowerCase()}`;
    this.btnPlay.disabled = this.status === "running";
    this.btnPause.disabled = this.status !== "running";
  }

  renderTree() {
    if (!this.treeData || !this.treeSvg) return;

    this.treeSvg.innerHTML = "";

    const nodes = [];
    const links = [];
    let nextX = 40;

    // Layout hierarchy nodes
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
      circle.addEventListener("click", () => this.selectNode(n));
      g.appendChild(circle);

      const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
      text.setAttribute("dy", 4);
      text.textContent = `N:${n.visits}`;
      g.appendChild(text);

      const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
      label.setAttribute("dy", 32);
      label.setAttribute("font-size", "10px");
      label.setAttribute("fill", "#94a3b8");
      const actStr = n.action ? n.action.slice(0, 15) : (n.depth === 0 ? "Root" : "");
      label.textContent = actStr;
      g.appendChild(label);

      this.treeSvg.appendChild(g);
    });

    if (!this.selectedNode && nodes.length > 0) {
      this.selectNode(nodes[0]);
    }
  }

  selectNode(node) {
    this.selectedNode = node;
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

    this.stateInspector.textContent = typeof node.state === "object"
      ? JSON.stringify(node.state, null, 2)
      : String(node.state);

    if (node.state && typeof node.state === "object" && node.state.stacks) {
      this.stateData = node.state;
      this.renderTabletop();
    }
  }

  renderTabletop() {
    if (!this.stacksArea) return;
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
