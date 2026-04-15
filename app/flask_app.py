"""Flask application for cPanel deployment."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from flask import Flask, jsonify, request, render_template_string
from src.pipeline.inference_pipeline import InferencePipeline, ModelNotFoundError
from src.data.ingestion import DataLoader

app = Flask(__name__)

inference = InferencePipeline(model_path=str(ROOT / "models"))
data_loader = DataLoader(str(ROOT / "data"))

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>datapulse</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        *, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }
        :root {
            --bg: #fafafa; --surface: #fff; --border: #e5e5e5;
            --text: #171717; --text2: #525252; --text3: #a3a3a3;
            --accent: #2563eb; --accent-light: #eff6ff;
            --green: #16a34a; --green-bg: #f0fdf4;
            --red: #dc2626; --red-bg: #fef2f2;
            --amber: #d97706; --amber-bg: #fffbeb;
        }
        body { font-family: 'Inter', system-ui, sans-serif; background: var(--bg);
               color: var(--text); min-height: 100vh; font-size: 14px; line-height: 1.5; }
        a { color: var(--accent); text-decoration: none; }

        /* layout */
        .shell { display: flex; min-height: 100vh; }
        nav { width: 220px; background: var(--surface); border-right: 1px solid var(--border);
              padding: 20px 0; position: fixed; top: 0; left: 0; bottom: 0; display: flex;
              flex-direction: column; }
        nav .logo { padding: 0 20px 20px; font-family: 'JetBrains Mono', monospace;
                    font-weight: 600; font-size: 15px; color: var(--text);
                    border-bottom: 1px solid var(--border); }
        nav .logo span { color: var(--accent); }
        nav ul { list-style: none; padding: 12px 8px; flex: 1; }
        nav li a { display: flex; align-items: center; gap: 10px; padding: 8px 12px;
                   border-radius: 6px; color: var(--text2); font-size: 13px; font-weight: 500;
                   transition: background 0.15s; }
        nav li a:hover { background: var(--bg); color: var(--text); }
        nav li a.active { background: var(--accent-light); color: var(--accent); }
        nav li a svg { width: 16px; height: 16px; flex-shrink: 0; }
        nav li a .kbd { margin-left: auto; font-size: 10px; color: var(--text3);
                        font-family: 'JetBrains Mono', monospace; background: var(--bg);
                        padding: 1px 5px; border-radius: 3px; border: 1px solid var(--border); }
        nav .nav-footer { padding: 12px 20px; border-top: 1px solid var(--border);
                          font-size: 11px; color: var(--text3); }
        nav .nav-footer .dot { width: 6px; height: 6px; background: var(--green);
                               border-radius: 50%; display: inline-block; margin-right: 4px; }

        .content { margin-left: 220px; flex: 1; padding: 32px 40px; max-width: 860px; }
        .content h2 { font-size: 18px; font-weight: 600; margin-bottom: 4px; }
        .content .subtitle { color: var(--text3); font-size: 13px; margin-bottom: 24px; }

        .page { display: none; }
        .page.active { display: block; }

        /* forms */
        .fields { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin-bottom: 20px; }
        .field label { display: block; font-size: 12px; font-weight: 500; color: var(--text2);
                       margin-bottom: 4px; }
        .field input[type=number], .field select { width: 100%; padding: 8px 10px;
                border: 1px solid var(--border); border-radius: 6px; font-size: 13px;
                font-family: inherit; background: var(--surface); color: var(--text);
                transition: border-color 0.15s, box-shadow 0.15s; outline: none; }
        .field input:focus { border-color: var(--accent);
                             box-shadow: 0 0 0 3px rgba(37,99,235,0.1); }
        .field.span2 { grid-column: span 2; }
        .field.span3 { grid-column: span 3; }

        .range-wrap { position: relative; }
        .range-wrap input[type=range] { -webkit-appearance: none; width: 100%; height: 4px;
                background: var(--border); border-radius: 2px; outline: none; margin-top: 8px; }
        .range-wrap input[type=range]::-webkit-slider-thumb { -webkit-appearance: none;
                width: 16px; height: 16px; background: var(--accent); border-radius: 50%;
                cursor: pointer; border: 2px solid white; box-shadow: 0 1px 3px rgba(0,0,0,0.15); }
        .range-val { position: absolute; right: 0; top: -2px; font-family: 'JetBrains Mono', monospace;
                     font-size: 13px; font-weight: 500; color: var(--accent); }

        .run-btn { display: inline-flex; align-items: center; gap: 6px; padding: 9px 20px;
                   background: var(--accent); color: white; border: none; border-radius: 6px;
                   font-size: 13px; font-weight: 500; font-family: inherit; cursor: pointer;
                   transition: background 0.15s, transform 0.1s; }
        .run-btn:hover { background: #1d4ed8; }
        .run-btn:active { transform: scale(0.98); }
        .run-btn:disabled { opacity: 0.5; cursor: wait; }
        .run-btn .spinner { width: 14px; height: 14px; border: 2px solid rgba(255,255,255,0.3);
                            border-top-color: white; border-radius: 50%; display: none;
                            animation: spin 0.6s linear infinite; }
        .run-btn.loading .spinner { display: inline-block; }
        .run-btn.loading .btn-text { display: none; }
        @keyframes spin { to { transform: rotate(360deg); } }

        /* results */
        .output { margin-top: 24px; padding: 20px; background: var(--surface);
                  border: 1px solid var(--border); border-radius: 8px; display: none; }
        .output.show { display: block; animation: fadeUp 0.25s ease; }
        @keyframes fadeUp { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }

        .big-num { font-size: 36px; font-weight: 600; font-family: 'JetBrains Mono', monospace;
                   letter-spacing: -1px; line-height: 1; }
        .big-num small { font-size: 14px; font-weight: 400; color: var(--text3); margin-left: 4px; }
        .metric-row { display: flex; gap: 32px; margin-top: 16px; padding-top: 16px;
                      border-top: 1px solid var(--border); }
        .metric { flex: 1; }
        .metric dt { font-size: 11px; color: var(--text3); font-weight: 500; margin-bottom: 2px; }
        .metric dd { font-size: 15px; font-weight: 600; font-family: 'JetBrains Mono', monospace; }

        .pill { display: inline-block; padding: 2px 10px; border-radius: 99px; font-size: 12px;
                font-weight: 600; }
        .pill-green { background: var(--green-bg); color: var(--green); }
        .pill-red { background: var(--red-bg); color: var(--red); }
        .pill-amber { background: var(--amber-bg); color: var(--amber); }

        /* probability bars */
        .prob-bars { display: flex; flex-direction: column; gap: 8px; margin-top: 16px; }
        .prob-bar { display: flex; align-items: center; gap: 10px; }
        .prob-bar .bar-label { width: 60px; font-size: 12px; color: var(--text2); font-weight: 500; }
        .prob-bar .bar-track { flex: 1; height: 20px; background: var(--bg); border-radius: 4px;
                               overflow: hidden; position: relative; }
        .prob-bar .bar-fill { height: 100%; border-radius: 4px; transition: width 0.5s ease;
                              display: flex; align-items: center; padding-left: 8px;
                              font-size: 11px; font-weight: 600; color: white; min-width: fit-content; }
        .bar-fill.low { background: var(--green); }
        .bar-fill.med { background: var(--amber); }
        .bar-fill.high { background: var(--red); }

        /* market gauge */
        .ev-display { display: flex; align-items: baseline; gap: 12px; }
        .ev-display .verdict { font-size: 13px; font-weight: 600; }

        /* datasets */
        .ds-tabs { display: flex; gap: 0; border-bottom: 1px solid var(--border); margin-bottom: 20px; }
        .ds-tab { padding: 8px 16px; font-size: 13px; color: var(--text3); cursor: pointer;
                  border-bottom: 2px solid transparent; font-weight: 500; transition: all 0.15s; }
        .ds-tab:hover { color: var(--text); }
        .ds-tab.active { color: var(--accent); border-bottom-color: var(--accent); }
        .ds-info { display: flex; gap: 24px; margin-bottom: 16px; }
        .ds-info span { font-size: 12px; color: var(--text3); }
        .ds-info strong { color: var(--text); }

        table { width: 100%; border-collapse: collapse; font-size: 12px; }
        table th { text-align: left; padding: 8px 10px; font-weight: 500; color: var(--text3);
                   border-bottom: 1px solid var(--border); background: var(--bg);
                   position: sticky; top: 0; font-size: 11px; }
        table td { padding: 7px 10px; border-bottom: 1px solid #f5f5f5; color: var(--text2);
                   font-family: 'JetBrains Mono', monospace; font-size: 12px; }
        table tr:hover td { background: #fafbff; }
        .table-wrap { max-height: 400px; overflow: auto; border: 1px solid var(--border);
                      border-radius: 8px; }

        /* toast */
        .toast { position: fixed; bottom: 24px; right: 24px; background: var(--text);
                 color: white; padding: 10px 16px; border-radius: 8px; font-size: 13px;
                 transform: translateY(100px); opacity: 0; transition: all 0.3s ease;
                 z-index: 100; font-weight: 500; }
        .toast.show { transform: none; opacity: 1; }

        .empty { text-align: center; padding: 40px; color: var(--text3); }

        @media (max-width: 768px) {
            nav { display: none; }
            .content { margin-left: 0; padding: 20px; }
            .fields { grid-template-columns: 1fr 1fr; }
            .field.span2, .field.span3 { grid-column: span 2; }
            .mobile-nav { display: flex; gap: 0; overflow-x: auto; border-bottom: 1px solid var(--border);
                          background: var(--surface); padding: 0 12px; }
            .mobile-nav a { padding: 10px 14px; font-size: 13px; white-space: nowrap;
                            color: var(--text3); border-bottom: 2px solid transparent; }
            .mobile-nav a.active { color: var(--accent); border-bottom-color: var(--accent); }
        }
        @media (min-width: 769px) { .mobile-nav { display: none; } }
    </style>
</head>
<body>
    <div class="mobile-nav">
        <a href="#player" class="active" onclick="go('player')">Players</a>
        <a href="#risk" onclick="go('risk')">Risk</a>
        <a href="#market" onclick="go('market')">Market</a>
        <a href="#data" onclick="go('data')">Data</a>
    </div>
    <div class="shell">
        <nav>
            <div class="logo"><span>data</span>pulse</div>
            <ul>
                <li><a href="#player" class="active" onclick="go('player')">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17 21v-2a4 4 0 00-4-4H5a4 4 0 00-4-4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 00-3-3.87"/><path d="M16 3.13a4 4 0 010 7.75"/></svg>
                    Players <span class="kbd">1</span>
                </a></li>
                <li><a href="#risk" onclick="go('risk')">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 9v2m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/></svg>
                    Risk <span class="kbd">2</span>
                </a></li>
                <li><a href="#market" onclick="go('market')">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 000 7h5a3.5 3.5 0 010 7H6"/></svg>
                    Market <span class="kbd">3</span>
                </a></li>
                <li><a href="#data" onclick="go('data')">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>
                    Data <span class="kbd">4</span>
                </a></li>
            </ul>
            <div class="nav-footer"><span class="dot"></span>MODELS_STATUS</div>
        </nav>

        <div class="content">
            <!-- PLAYER PAGE -->
            <div id="pg-player" class="page active">
                <h2>Player prediction</h2>
                <p class="subtitle">Estimate fantasy points from player stats. Uses a gradient boosting model trained on historical data.</p>
                <div class="fields">
                    <div class="field">
                        <label>Points avg</label>
                        <input type="number" id="points_avg" value="20" step="0.1">
                    </div>
                    <div class="field">
                        <label>Assists avg</label>
                        <input type="number" id="assists_avg" value="5" step="0.1">
                    </div>
                    <div class="field">
                        <label>Rebounds avg</label>
                        <input type="number" id="rebounds_avg" value="7" step="0.1">
                    </div>
                    <div class="field">
                        <label>Games played</label>
                        <input type="number" id="games_played" value="60">
                    </div>
                    <div class="field">
                        <label>Steals avg</label>
                        <input type="number" id="steals_avg" value="1.2" step="0.1">
                    </div>
                    <div class="field">
                        <label>Blocks avg</label>
                        <input type="number" id="blocks_avg" value="0.8" step="0.1">
                    </div>
                    <div class="field">
                        <label>Turnovers avg</label>
                        <input type="number" id="turnovers_avg" value="2.0" step="0.1">
                    </div>
                    <div class="field">
                        <label>FG%</label>
                        <input type="number" id="fg_pct" value="0.48" step="0.01">
                    </div>
                    <div class="field">
                        <label>Salary</label>
                        <input type="number" id="salary" value="7000" step="100">
                    </div>
                    <div class="field span3">
                        <label>Consistency</label>
                        <div class="range-wrap">
                            <span class="range-val" id="con-val">0.70</span>
                            <input type="range" id="consistency_score" min="0" max="1" step="0.01" value="0.7"
                                   oninput="document.getElementById('con-val').textContent=Number(this.value).toFixed(2)">
                        </div>
                    </div>
                </div>
                <button class="run-btn" onclick="predictPlayer(this)">
                    <span class="spinner"></span>
                    <span class="btn-text">Run prediction</span>
                </button>
                <div id="player-out" class="output"></div>
            </div>

            <!-- RISK PAGE -->
            <div id="pg-risk" class="page">
                <h2>Risk scoring</h2>
                <p class="subtitle">Classify a user's risk profile based on their DFS activity. Random forest classifier with 3 tiers.</p>
                <div class="fields">
                    <div class="field span3">
                        <label>Win rate</label>
                        <div class="range-wrap">
                            <span class="range-val" id="wr-val">0.50</span>
                            <input type="range" id="win_rate" min="0" max="1" step="0.01" value="0.5"
                                   oninput="document.getElementById('wr-val').textContent=Number(this.value).toFixed(2)">
                        </div>
                    </div>
                    <div class="field">
                        <label>Total wagered ($)</label>
                        <input type="number" id="total_wagered" value="10000">
                    </div>
                    <div class="field">
                        <label>Total entries</label>
                        <input type="number" id="total_entries" value="500">
                    </div>
                    <div class="field">
                        <label>ROI</label>
                        <input type="number" id="roi" value="0.05" step="0.01">
                    </div>
                </div>
                <button class="run-btn" onclick="scoreRisk(this)">
                    <span class="spinner"></span>
                    <span class="btn-text">Score risk</span>
                </button>
                <div id="risk-out" class="output"></div>
            </div>

            <!-- MARKET PAGE -->
            <div id="pg-market" class="page">
                <h2>Market evaluation</h2>
                <p class="subtitle">Check if a line has positive expected value. Plug in the American odds and your probability estimate.</p>
                <div class="fields">
                    <div class="field">
                        <label>American odds</label>
                        <input type="number" id="odds" value="-110" placeholder="-110, +150, etc.">
                    </div>
                    <div class="field span2">
                        <label>Your estimated probability</label>
                        <div class="range-wrap">
                            <span class="range-val" id="prob-val">0.55</span>
                            <input type="range" id="est_prob" min="0.01" max="0.99" step="0.01" value="0.55"
                                   oninput="document.getElementById('prob-val').textContent=Number(this.value).toFixed(2)">
                        </div>
                    </div>
                </div>
                <button class="run-btn" onclick="evalMarket(this)">
                    <span class="spinner"></span>
                    <span class="btn-text">Evaluate</span>
                </button>
                <div id="market-out" class="output"></div>
            </div>

            <!-- DATA PAGE -->
            <div id="pg-data" class="page">
                <h2>Data explorer</h2>
                <p class="subtitle">Browse the sample datasets used for training and evaluation.</p>
                <div id="ds-tabs" class="ds-tabs"></div>
                <div id="ds-content">
                    <div class="empty">Loading datasets...</div>
                </div>
            </div>
        </div>
    </div>

    <div class="toast" id="toast"></div>

<script>
// nav + routing
function go(page) {
    document.querySelectorAll('nav a, .mobile-nav a').forEach(a => a.classList.remove('active'));
    document.querySelectorAll(`nav a[href="#${page}"], .mobile-nav a[href="#${page}"]`).forEach(a => a.classList.add('active'));
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.getElementById('pg-' + page).classList.add('active');
    location.hash = page;
    if (page === 'data' && !dsLoaded) loadDatasets();
}

// keyboard shortcuts
document.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT') return;
    if (e.key === '1') go('player');
    if (e.key === '2') go('risk');
    if (e.key === '3') go('market');
    if (e.key === '4') go('data');
});

// handle initial hash
window.addEventListener('load', () => {
    const h = location.hash.slice(1);
    if (['player','risk','market','data'].includes(h)) go(h);
});

// enter key submits
document.addEventListener('keydown', e => {
    if (e.key !== 'Enter' || e.target.tagName !== 'INPUT') return;
    const page = document.querySelector('.page.active').id.replace('pg-','');
    const btn = document.querySelector('.page.active .run-btn');
    if (btn && !btn.disabled) btn.click();
});

function toast(msg) {
    const t = document.getElementById('toast');
    t.textContent = msg;
    t.classList.add('show');
    setTimeout(() => t.classList.remove('show'), 2500);
}

async function api(url, body) {
    const opts = body
        ? { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body) }
        : {};
    const res = await fetch(url, opts);
    if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error || `${res.status} ${res.statusText}`);
    }
    return res.json();
}

function btnLoad(btn, on) {
    btn.disabled = on;
    btn.classList.toggle('loading', on);
}

// -- player --
async function predictPlayer(btn) {
    btnLoad(btn, true);
    const ids = ['games_played','points_avg','assists_avg','rebounds_avg','steals_avg',
                 'blocks_avg','turnovers_avg','fg_pct','salary','consistency_score'];
    const features = {};
    ids.forEach(id => features[id] = parseFloat(document.getElementById(id).value));

    try {
        const r = await api('/api/predict/player', { features });
        const pts = r.prediction;
        const sal = features.salary;
        const vpk = sal > 0 ? (pts / (sal / 1000)) : 0;
        const rating = vpk > 5 ? 'Excellent' : vpk > 3 ? 'Good' : 'Below avg';
        const rCls = vpk > 5 ? 'green' : vpk > 3 ? 'amber' : 'red';

        const el = document.getElementById('player-out');
        el.innerHTML = `
            <div class="big-num">${pts.toFixed(1)}<small>fantasy pts</small></div>
            <dl class="metric-row">
                <div class="metric">
                    <dt>Value / $1k salary</dt>
                    <dd style="color:var(--${rCls})">${vpk.toFixed(2)}</dd>
                </div>
                <div class="metric">
                    <dt>Rating</dt>
                    <dd><span class="pill pill-${rCls}">${rating}</span></dd>
                </div>
                <div class="metric">
                    <dt>Model</dt>
                    <dd style="color:var(--text3)">${r.model_version}</dd>
                </div>
            </dl>`;
        el.classList.add('show');
        toast('Prediction complete');
    } catch(e) {
        toast('Error: ' + e.message);
    }
    btnLoad(btn, false);
}

// -- risk --
async function scoreRisk(btn) {
    btnLoad(btn, true);
    const wr = parseFloat(document.getElementById('win_rate').value);
    const tw = parseFloat(document.getElementById('total_wagered').value);
    const te = parseFloat(document.getElementById('total_entries').value);
    const roi = parseFloat(document.getElementById('roi').value);
    const features = {
        total_entries: te, win_rate: wr, avg_entry_fee: tw / Math.max(te, 1),
        total_wagered: tw, total_won: tw * (1 + roi), net_profit: tw * roi,
    };

    try {
        const r = await api('/api/predict/risk', { features });
        const labels = ['Low', 'Medium', 'High'];
        const cls = ['green', 'amber', 'red'];
        const cat = r.risk_category;
        const probs = r.probabilities;

        const el = document.getElementById('risk-out');
        el.innerHTML = `
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px;">
                <span class="pill pill-${cls[cat]}" style="font-size:14px;padding:4px 14px;">${labels[cat]} risk</span>
                <span style="color:var(--text3);font-size:13px;">${(Math.max(...probs)*100).toFixed(0)}% confidence</span>
            </div>
            <div class="prob-bars">
                <div class="prob-bar">
                    <span class="bar-label">Low</span>
                    <div class="bar-track"><div class="bar-fill low" style="width:${(probs[0]*100).toFixed(1)}%">${(probs[0]*100).toFixed(1)}%</div></div>
                </div>
                <div class="prob-bar">
                    <span class="bar-label">Medium</span>
                    <div class="bar-track"><div class="bar-fill med" style="width:${(probs[1]*100).toFixed(1)}%">${(probs[1]*100).toFixed(1)}%</div></div>
                </div>
                <div class="prob-bar">
                    <span class="bar-label">High</span>
                    <div class="bar-track"><div class="bar-fill high" style="width:${(probs[2]*100).toFixed(1)}%">${(probs[2]*100).toFixed(1)}%</div></div>
                </div>
            </div>`;
        el.classList.add('show');
        toast('Risk scored');
    } catch(e) {
        toast('Error: ' + e.message);
    }
    btnLoad(btn, false);
}

// -- market --
async function evalMarket(btn) {
    btnLoad(btn, true);
    const odds = parseFloat(document.getElementById('odds').value);
    const prob = parseFloat(document.getElementById('est_prob').value);

    try {
        const r = await api('/api/market/evaluate', { odds, estimated_probability: prob });
        const pos = r.expected_value > 0;
        const edgePos = r.edge > 0;

        const el = document.getElementById('market-out');
        el.innerHTML = `
            <div class="ev-display">
                <div class="big-num" style="color:var(--${pos ? 'green' : 'red'})">${pos ? '+' : ''}${r.expected_value.toFixed(3)}<small>EV</small></div>
                <span class="pill pill-${r.recommendation === 'Bet' ? 'green' : 'red'}" style="font-size:14px;padding:4px 14px;">${r.recommendation === 'Bet' ? 'Take it' : 'Pass'}</span>
            </div>
            <dl class="metric-row">
                <div class="metric">
                    <dt>Implied probability</dt>
                    <dd>${(r.implied_probability * 100).toFixed(1)}%</dd>
                </div>
                <div class="metric">
                    <dt>Your estimate</dt>
                    <dd>${(prob * 100).toFixed(1)}%</dd>
                </div>
                <div class="metric">
                    <dt>Edge</dt>
                    <dd style="color:var(--${edgePos ? 'green' : 'red'})">${edgePos ? '+' : ''}${(r.edge * 100).toFixed(1)}%</dd>
                </div>
            </dl>`;
        el.classList.add('show');
        toast(r.recommendation === 'Bet' ? 'Positive EV found' : 'Negative EV');
    } catch(e) {
        toast('Error: ' + e.message);
    }
    btnLoad(btn, false);
}

// -- data --
let dsLoaded = false;
async function loadDatasets() {
    dsLoaded = true;
    try {
        const r = await api('/api/datasets');
        const tabs = document.getElementById('ds-tabs');
        tabs.innerHTML = r.datasets.map((d, i) =>
            `<div class="ds-tab ${i===0?'active':''}" onclick="loadDS('${d}',this)">${d.replace('_',' ')}</div>`
        ).join('');
        if (r.datasets.length) loadDS(r.datasets[0], tabs.children[0]);
    } catch(e) {
        document.getElementById('ds-content').innerHTML = `<div class="empty">Failed to load: ${e.message}</div>`;
    }
}

async function loadDS(name, tab) {
    document.querySelectorAll('.ds-tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById('ds-content').innerHTML = '<div class="empty">Loading...</div>';

    try {
        const r = await api('/api/dataset/' + name);
        const cols = r.sample[0] ? Object.keys(r.sample[0]) : [];
        let html = `
            <div class="ds-info">
                <span><strong>${r.rows.toLocaleString()}</strong> rows</span>
                <span><strong>${r.columns}</strong> columns</span>
                <span><strong>${r.numeric_columns}</strong> numeric</span>
            </div>
            <div class="table-wrap"><table>
                <thead><tr>${cols.map(c => `<th>${c}</th>`).join('')}</tr></thead>
                <tbody>`;
        r.sample.forEach(row => {
            html += '<tr>' + cols.map(c => {
                let v = row[c];
                if (typeof v === 'number') v = Number.isInteger(v) ? v.toLocaleString() : v.toFixed(3);
                return `<td>${v}</td>`;
            }).join('') + '</tr>';
        });
        html += '</tbody></table></div>';
        document.getElementById('ds-content').innerHTML = html;
    } catch(e) {
        document.getElementById('ds-content').innerHTML = `<div class="empty">${e.message}</div>`;
    }
}
</script>
</body>
</html>
"""


@app.route('/')
def index():
    models = ', '.join(inference.available_models()) or 'none'
    return render_template_string(HTML_TEMPLATE.replace('MODELS_STATUS', models))


@app.route('/api/health')
def health():
    return jsonify({
        "status": "healthy",
        "version": "1.0.0",
        "models_loaded": inference.available_models(),
    })


@app.route('/api/predict/player', methods=['POST'])
def predict_player():
    data = request.get_json() or {}
    features = data.get('features', {})
    try:
        result = inference.predict_player(features)
        return jsonify(result)
    except ModelNotFoundError as e:
        return jsonify({"error": str(e)}), 503


@app.route('/api/predict/risk', methods=['POST'])
def predict_risk():
    data = request.get_json() or {}
    features = data.get('features', {})
    try:
        result = inference.score_risk(features)
        return jsonify(result)
    except ModelNotFoundError as e:
        return jsonify({"error": str(e)}), 503


@app.route('/api/market/evaluate', methods=['POST'])
def evaluate_market():
    data = request.get_json() or {}
    odds = data.get('odds', -110)
    prob = data.get('estimated_probability', 0.55)
    implied = 100 / (odds + 100) if odds > 0 else abs(odds) / (abs(odds) + 100)
    payout = odds / 100 if odds > 0 else 100 / abs(odds)
    ev = (prob * payout) - (1 - prob)
    return jsonify({
        "implied_probability": round(implied, 4),
        "expected_value": round(ev, 4),
        "edge": round(prob - implied, 4),
        "recommendation": "Bet" if ev > 0.02 else "Pass",
    })


@app.route('/api/datasets')
def list_datasets():
    available = data_loader.list_available("sample")
    return jsonify({"datasets": available, "count": len(available)})


@app.route('/api/dataset/<name>')
def get_dataset(name):
    try:
        df = data_loader.get_sample_data(name)
    except FileNotFoundError:
        return jsonify({"error": "Dataset not found"}), 404
    numeric = df.select_dtypes(include='number').columns.tolist()
    return jsonify({
        "rows": len(df),
        "columns": len(df.columns),
        "numeric_columns": len(numeric),
        "column_names": df.columns.tolist(),
        "sample": df.head(10).fillna("").to_dict(orient="records"),
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
