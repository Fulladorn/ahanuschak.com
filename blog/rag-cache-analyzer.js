/* RAG vs Cache-Augmented Generation — Interactive Cost Analyzer
   Self-contained IIFE. Exposes window.initRagCacheWidgets().
   Mounts into .rag-cache-widget-placeholder elements after post render. */
(function () {
    'use strict';

    // =========================================================
    // PRICING DATA  (all prices in $ per million tokens)
    // =========================================================
    var LLM_MODELS = {
        sonnet46: { label: 'Claude Sonnet 4.6',  inp: 3,    out: 15,  cw5: 3.75,  cr: 0.30  },
        opus46:   { label: 'Claude Opus 4.6',    inp: 15,   out: 75,  cw5: 18.75, cr: 1.50  },
        haiku45:  { label: 'Claude Haiku 4.5',   inp: 1,    out: 5,   cw5: 1.25,  cr: 0.10  },
        gpt52:    { label: 'GPT-5.2',            inp: 1.75, out: 14,  cw5: 1.75,  cr: 0.175 },
        gemini3:  { label: 'Gemini 3 Pro',       inp: 2,    out: 12,  cw5: 2.00,  cr: 0.50  },
    };

    var EMBED_MODELS = {
        te3large: { label: 'OpenAI text-embedding-3-large', price: 0.13, dims: 1536 },
        te3small: { label: 'OpenAI text-embedding-3-small', price: 0.02, dims: 1536 },
        cohere4:  { label: 'Cohere Embed 4',                price: 0.12, dims: 1536 },
        voyage35: { label: 'Voyage AI voyage-3.5',          price: 0.06, dims: 1024 },
    };

    var VDB_OPTIONS = {
        pinecone_std: { label: 'Pinecone Standard',     ruPerM: 8.25,  wuPerM: 2.00, storPerGB: 0.33 },
        pinecone_ent: { label: 'Pinecone Enterprise',   ruPerM: 16,    wuPerM: 2.00, storPerGB: 0.33 },
        weaviate:     { label: 'Weaviate Cloud',        dimPerM: 0.095 },
        self_hosted:  { label: 'Self-hosted / pgvector' },
    };

    // =========================================================
    // DEFAULT STATE
    // =========================================================
    function defaultState() {
        return {
            llm:              'sonnet46',
            embed:            'te3large',
            stableTok:        237500,
            qTok:             250,
            sysTok:           500,
            cacheEnabled:     true,
            ttl:              '5min',
            cacheWriteEvents: 1,
            topN:             25,
            topK:             8,
            chunkSize:        250,
            vdb:              'pinecone_std',
            nCalls:           100,
            outTok:           500,
            activeTab:        0,
        };
    }

    // =========================================================
    // CALCULATIONS
    // =========================================================
    function derive(s) {
        var embed  = EMBED_MODELS[s.embed];
        var llm    = LLM_MODELS[s.llm];
        var vdb    = VDB_OPTIONS[s.vdb];

        var chunks        = Math.ceil(s.stableTok / s.chunkSize);
        var storageGB     = (chunks * embed.dims * 4 * 1.5) / 1073741824;
        var effectiveTopN = Math.min(s.topN, chunks);
        var effectiveTopK = Math.min(s.topK, effectiveTopN);
        var ragLLMInput   = s.sysTok + (effectiveTopK * s.chunkSize) + s.qTok;
        var cacheLLMInput = s.stableTok + s.qTok;

        // ---- RAG costs ----
        var embCorpus  = (s.stableTok / 1e6) * embed.price;
        var embQuery   = s.nCalls * (s.qTok / 1e6) * embed.price;
        var rerankCost = s.nCalls * Math.ceil(effectiveTopN / 100) * 0.002;

        var vdbCost;
        if (s.vdb === 'pinecone_std' || s.vdb === 'pinecone_ent') {
            vdbCost = storageGB * vdb.storPerGB
                    + (chunks / 1e6) * vdb.wuPerM
                    + s.nCalls * Math.max(0.25, storageGB) * (vdb.ruPerM / 1e6);
        } else if (s.vdb === 'weaviate') {
            vdbCost = (chunks * embed.dims / 1e6) * 0.095;
        } else {
            vdbCost = 0;
        }

        var ragLLMInpCost = s.nCalls * (ragLLMInput / 1e6) * llm.inp;
        var ragLLMOutCost = s.nCalls * (s.outTok / 1e6) * llm.out;
        var ragTotal      = embCorpus + embQuery + rerankCost + vdbCost + ragLLMInpCost + ragLLMOutCost;

        // ---- Cache-aug costs ----
        var outCost = s.nCalls * (s.outTok / 1e6) * llm.out;
        var cacheWrite = 0, cacheRead = 0, freshInp = 0, cacheTotal;

        if (s.cacheEnabled) {
            var cacheWriteRate = (s.ttl === '5min') ? llm.cw5 : llm.cw5 * 2;
            cacheWrite = s.cacheWriteEvents * (s.stableTok / 1e6) * cacheWriteRate;
            cacheRead  = Math.max(0, s.nCalls - s.cacheWriteEvents) * (s.stableTok / 1e6) * llm.cr;
            freshInp   = s.nCalls * (s.qTok / 1e6) * llm.inp;
            cacheTotal = cacheWrite + cacheRead + freshInp + outCost;
        } else {
            cacheTotal = s.nCalls * ((s.stableTok + s.qTok) / 1e6) * llm.inp + outCost;
        }

        return {
            chunks, storageGB, effectiveTopN, effectiveTopK,
            ragLLMInput, cacheLLMInput,
            embCorpus, embQuery, rerankCost, vdbCost,
            ragLLMInpCost, ragLLMOutCost, ragTotal,
            cacheWrite, cacheRead, freshInp, outCost, cacheTotal,
        };
    }

    // =========================================================
    // FORMATTING HELPERS
    // =========================================================
    function fmt(n) {
        if (n === 0) return '$0.00';
        if (n < 0.001) return '<$0.001';
        if (n < 1) return '$' + n.toFixed(3);
        if (n < 100) return '$' + n.toFixed(2);
        return '$' + n.toFixed(2).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
    }
    function fmtN(n) {
        return Number(n).toLocaleString();
    }
    function fmtGB(n) {
        if (n < 1 / (1024 * 1024)) return (n * 1024 * 1024 * 1024).toFixed(0) + ' B';
        if (n < 1 / 1024) return (n * 1024 * 1024).toFixed(1) + ' KB';
        if (n < 1) return (n * 1024).toFixed(1) + ' MB';
        return n.toFixed(3) + ' GB';
    }
    function fmtTok(n) {
        if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M';
        if (n >= 1e3) return (n / 1e3).toFixed(0) + 'K';
        return String(n);
    }
    function esc(s) {
        return String(s)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    // =========================================================
    // HTML BUILDERS
    // =========================================================
    function selectOpts(map, current) {
        return Object.keys(map).map(function (k) {
            return '<option value="' + k + '"' + (k === current ? ' selected' : '') + '>'
                + esc(map[k].label) + '</option>';
        }).join('');
    }

    function sliderField(name, label, value, min, max, step) {
        return '<div class="rca-field rca-field-slider">'
            + '<label class="rca-label">' + label
            + ' <span class="rca-slider-val" data-for="' + name + '">' + fmtN(value) + '</span></label>'
            + '<input type="range" class="rca-range" data-bind="' + name + '"'
            + ' min="' + min + '" max="' + max + '" step="' + step + '" value="' + value + '">'
            + '</div>';
    }

    function pill(label, value) {
        return '<div class="rca-pill">'
            + '<span class="rca-pill-label">' + esc(label) + '</span>'
            + '<span class="rca-pill-value">' + esc(value) + '</span>'
            + '</div>';
    }

    function summaryCard(title, value, sub, extraClass) {
        return '<div class="rca-card' + (extraClass ? ' ' + extraClass : '') + '">'
            + '<div class="rca-card-title">' + esc(title) + '</div>'
            + '<div class="rca-card-value">' + esc(value) + '</div>'
            + '<div class="rca-card-sub">' + esc(sub) + '</div>'
            + '</div>';
    }

    function buildBreakdown(s, d) {
        var ragLines = [
            { label: 'Corpus Embeddings (one-time)', value: d.embCorpus },
            { label: 'Query Embeddings',             value: d.embQuery },
            { label: 'Reranker',                     value: d.rerankCost },
            { label: 'Vector Database',              value: d.vdbCost },
            { label: 'LLM Input',                    value: d.ragLLMInpCost },
            { label: 'LLM Output',                   value: d.ragLLMOutCost },
        ];

        var cacheLines;
        if (s.cacheEnabled) {
            cacheLines = [
                { label: 'Cache Writes (' + s.ttl + ' TTL)', value: d.cacheWrite },
                { label: 'Cache Reads',                       value: d.cacheRead },
                { label: 'Fresh Input (query tokens)',         value: d.freshInp },
                { label: 'LLM Output',                        value: d.outCost },
            ];
        } else {
            cacheLines = [
                { label: 'Full Context Input (no cache)', value: d.cacheTotal - d.outCost },
                { label: 'LLM Output',                    value: d.outCost },
            ];
        }

        function rows(lines) {
            return lines.map(function (l) {
                return '<tr><td>' + esc(l.label) + '</td>'
                    + '<td class="rca-bd-val">' + fmt(l.value) + '</td></tr>';
            }).join('');
        }

        return '<div class="rca-breakdown">'
            + '<div class="rca-bd-col">'
            +   '<div class="rca-bd-head">RAG</div>'
            +   '<table class="rca-bd-table"><tbody>' + rows(ragLines) + '</tbody>'
            +   '<tfoot><tr><td><strong>Total</strong></td>'
            +   '<td class="rca-bd-val"><strong>' + fmt(d.ragTotal) + '</strong></td></tr></tfoot>'
            +   '</table>'
            + '</div>'
            + '<div class="rca-bd-col">'
            +   '<div class="rca-bd-head">Cache-Augmented</div>'
            +   '<table class="rca-bd-table"><tbody>' + rows(cacheLines) + '</tbody>'
            +   '<tfoot><tr><td><strong>Total</strong></td>'
            +   '<td class="rca-bd-val"><strong>' + fmt(d.cacheTotal) + '</strong></td></tr></tfoot>'
            +   '</table>'
            + '</div>'
            + '</div>';
    }

    function buildAssumptions(s, d) {
        var rerankCalls = Math.ceil(d.effectiveTopN / 100);
        return '<ul class="rca-assumptions">'
            + '<li>Reranker priced at <strong>$2/1,000 searches</strong> (Cohere Rerank API). '
            +   'Each query uses <strong>' + rerankCalls + '</strong> search call(s) '
            +   '(max 100 docs/call), totalling '
            +   '<strong>' + fmtN(rerankCalls * s.nCalls) + '</strong> calls for this batch.</li>'
            + '<li>Pinecone read units: 1 RU per query per GB of namespace (min 0.25 GB billed).</li>'
            + '<li>Vector storage estimated at <strong>' + fmtGB(d.storageGB) + '</strong> '
            +   '(chunks × dims × 4 bytes × 1.5× overhead).</li>'
            + '<li>Corpus embedding is a <strong>one-time cost</strong>; '
            +   'query embeddings recur every call.</li>'
            + '<li>Cache TTL "5 min" applies the standard 1× write premium; '
            +   '"1 hr" applies a 2× premium.</li>'
            + '<li>GPT-5.2 uses <strong>automatic caching</strong> — '
            +   'no explicit write cost; write price equals input price.</li>'
            + '<li>Gemini 3 Pro uses <strong>implicit caching</strong> — '
            +   'cache write price equals input price; assumes all stable tokens are cached.</li>'
            + '<li>Output tokens fixed at <strong>' + fmtN(s.outTok) + ' tokens/call</strong> '
            +   '— adjust for your workload.</li>'
            + '<li>All costs are <strong>per-batch</strong> ('
            +   fmtN(s.nCalls) + ' LLM calls). '
            +   'The "Cost at Scale" tab shows cost across different volumes.</li>'
            + '<li>Self-hosted / pgvector VDB cost is $0 — assumes no additional infrastructure overhead.</li>'
            + '</ul>';
    }

    function buildWidget(s) {
        var d = derive(s);
        var cacheWins = d.cacheTotal <= d.ragTotal;
        var cheaper   = Math.abs(d.ragTotal - d.cacheTotal);
        var savings   = (d.ragTotal > 0)
            ? (cheaper / Math.max(d.ragTotal, d.cacheTotal)) * 100
            : 0;

        var ctxAlert = (s.stableTok > 200000 && (s.llm === 'sonnet46' || s.llm === 'opus46'))
            ? '<div class="rca-alert">&#9888; Stable context ('
              + fmtTok(s.stableTok) + ' tok) exceeds the 200K recommended limit for '
              + esc(LLM_MODELS[s.llm].label)
              + '. Consider chunking or switching models.</div>'
            : '';

        var topNAlert = s.topN > d.chunks
            ? '<div class="rca-alert">&#9888; topN (' + s.topN + ') exceeds corpus chunks ('
              + d.chunks + '). Clamped to ' + d.effectiveTopN + '.</div>'
            : '';

        var rerankInfo = '<div class="rca-alert rca-alert-info">'
            + '&#8505; Reranker: '
            + Math.ceil(d.effectiveTopN / 100) + ' search call(s) per query &times; '
            + s.nCalls + ' queries = '
            + fmtN(Math.ceil(d.effectiveTopN / 100) * s.nCalls)
            + ' total. Priced at $2 / 1,000 searches (Cohere Rerank API).</div>';

        var tabLabels = ['Breakdown', 'Bar Chart', 'Cost at Scale', 'Assumptions'];
        var tabBtns = tabLabels.map(function (lbl, i) {
            return '<button class="rca-tab-btn' + (s.activeTab === i ? ' rca-tab-active' : '')
                + '" data-tab="' + i + '">' + lbl + '</button>';
        }).join('');

        var winnerLabel = cacheWins ? 'Cache-Aug Wins' : 'RAG Wins';
        var winnerSub   = cacheWins
            ? 'saves ' + fmt(cheaper) + ' vs RAG'
            : 'saves ' + fmt(cheaper) + ' vs Cache-Aug';

        var panels = [
            buildBreakdown(s, d),
            '<div class="rca-chart-wrap"><canvas class="rca-bar-canvas"></canvas></div>',
            '<div class="rca-chart-wrap"><canvas class="rca-scale-canvas"></canvas></div>',
            buildAssumptions(s, d),
        ];

        var panelHtml = panels.map(function (p, i) {
            return '<div class="rca-panel' + (s.activeTab === i ? ' rca-panel-active' : '') + '">' + p + '</div>';
        }).join('');

        return '<div class="rca-widget">'

            // Header
            + '<div class="rca-header">'
            +   '<span class="section-label" style="font-size:0.7rem">interactive calculator</span>'
            +   '<h3 class="rca-title">RAG vs Cache-Augmented Generation — Cost Analyzer</h3>'
            + '</div>'

            // Model Selection
            + '<div class="rca-section">'
            +   '<div class="rca-section-title">Model Selection</div>'
            +   '<div class="rca-row">'
            +     '<div class="rca-field">'
            +       '<label class="rca-label">LLM</label>'
            +       '<select class="rca-select" data-bind="llm">' + selectOpts(LLM_MODELS, s.llm) + '</select>'
            +     '</div>'
            +     '<div class="rca-field">'
            +       '<label class="rca-label">Embedding Model</label>'
            +       '<select class="rca-select" data-bind="embed">' + selectOpts(EMBED_MODELS, s.embed) + '</select>'
            +     '</div>'
            +   '</div>'
            + '</div>'

            // Prompt Composition
            + '<div class="rca-section">'
            +   '<div class="rca-section-title">Prompt Composition</div>'
            +   sliderField('stableTok', 'Stable Context (tokens)', s.stableTok, 1000, 500000, 1000)
            +   sliderField('qTok',      'Query Tokens',            s.qTok,      50,   2000,   50)
            +   sliderField('sysTok',    'System Prompt Tokens',    s.sysTok,    0,    2000,   50)
            + '</div>'

            // Derived pills
            + '<div class="rca-derived">'
            +   pill('Corpus Chunks',    fmtN(d.chunks))
            +   pill('Vector Storage',   fmtGB(d.storageGB))
            +   pill('RAG LLM Input',    fmtTok(d.ragLLMInput) + ' tok')
            +   pill('Cache LLM Input',  fmtTok(d.cacheLLMInput) + ' tok')
            + '</div>'

            // Cache Settings
            + '<div class="rca-section">'
            +   '<div class="rca-section-title">Cache Settings</div>'
            +   '<div class="rca-row rca-row-wrap">'
            +     '<div class="rca-field">'
            +       '<label class="rca-label">Prompt Caching</label>'
            +       '<button class="rca-toggle' + (s.cacheEnabled ? ' rca-toggle-on' : '') + '" data-bind="cacheEnabled">'
            +         (s.cacheEnabled ? 'Enabled' : 'Disabled')
            +       '</button>'
            +     '</div>'
            +     '<div class="rca-field">'
            +       '<label class="rca-label">Cache TTL</label>'
            +       '<select class="rca-select" data-bind="ttl"' + (!s.cacheEnabled ? ' disabled' : '') + '>'
            +         '<option value="5min"' + (s.ttl === '5min' ? ' selected' : '') + '>5 minutes</option>'
            +         '<option value="1hr"' + (s.ttl === '1hr' ? ' selected' : '') + '>1 hour</option>'
            +       '</select>'
            +     '</div>'
            +     '<div class="rca-field">'
            +       '<label class="rca-label">Cache Write Events</label>'
            +       '<input class="rca-input" type="number" min="0" max="' + s.nCalls + '" value="'
            +         s.cacheWriteEvents + '" data-bind="cacheWriteEvents"'
            +         (!s.cacheEnabled ? ' disabled' : '') + '>'
            +     '</div>'
            +   '</div>'
            + '</div>'

            // RAG Retrieval
            + '<div class="rca-section">'
            +   '<div class="rca-section-title">RAG Retrieval</div>'
            +   '<div class="rca-row rca-row-wrap">'
            +     '<div class="rca-field">'
            +       '<label class="rca-label">Top-N Retrieved (pre-rerank)</label>'
            +       '<input class="rca-input" type="number" min="1" max="500" value="' + s.topN + '" data-bind="topN">'
            +     '</div>'
            +   '</div>'
            +   sliderField('topK',      'Top-K Injected into Prompt', s.topK,      1,  50,   1)
            +   sliderField('chunkSize', 'Chunk Size (tokens)',         s.chunkSize, 50, 1000, 50)
            +   '<div class="rca-field">'
            +     '<label class="rca-label">Vector Database</label>'
            +     '<select class="rca-select" data-bind="vdb">' + selectOpts(VDB_OPTIONS, s.vdb) + '</select>'
            +   '</div>'
            +   rerankInfo
            + '</div>'

            // Volume
            + '<div class="rca-section">'
            +   '<div class="rca-section-title">Volume</div>'
            +   sliderField('nCalls', 'Number of LLM Calls',      s.nCalls, 1,  1000, 1)
            +   sliderField('outTok', 'Output Tokens per Call',    s.outTok, 50, 4000, 50)
            + '</div>'

            // Alerts
            + '<div class="rca-alerts">' + ctxAlert + topNAlert + '</div>'

            // Summary cards
            + '<div class="rca-summary">'
            +   summaryCard('RAG Total',          fmt(d.ragTotal),   'Embed + VDB + Reranker + LLM',   '')
            +   summaryCard('Cache-Aug Total',     fmt(d.cacheTotal), s.cacheEnabled ? 'Cache write + read + LLM' : 'Full context (no cache)', '')
            +   summaryCard(
                    (cacheWins ? '\u2714 ' : '\u2714 ') + winnerLabel,
                    savings.toFixed(1) + '% cheaper',
                    winnerSub,
                    cacheWins ? 'rca-card-green' : 'rca-card-blue'
                )
            + '</div>'

            // Banner
            + '<div class="rca-banner ' + (cacheWins ? 'rca-banner-green' : 'rca-banner-blue') + '">'
            +   '<strong>' + (cacheWins ? 'Cache-Augmented Generation' : 'RAG') + '</strong>'
            +   ' is <strong>' + savings.toFixed(1) + '%</strong> cheaper for these parameters'
            +   ' &mdash; saves <strong>' + fmt(cheaper) + '</strong> per batch.'
            + '</div>'

            // Tabs
            + '<div class="rca-tabs">' + tabBtns + '</div>'
            + '<div class="rca-panels">' + panelHtml + '</div>'

            + '</div>'; // .rca-widget
    }

    // =========================================================
    // CHART.JS LAZY LOAD
    // =========================================================
    var CHART_CDN = 'https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js';
    var chartScriptState = 'idle'; // idle | loading | ready
    var chartQueue = [];

    function withChart(cb) {
        if (window.Chart) { cb(); return; }
        chartQueue.push(cb);
        if (chartScriptState !== 'idle') return;
        chartScriptState = 'loading';
        var s = document.createElement('script');
        s.src = CHART_CDN;
        s.onload = function () {
            chartScriptState = 'ready';
            var q = chartQueue.slice();
            chartQueue = [];
            q.forEach(function (fn) { fn(); });
        };
        document.head.appendChild(s);
    }

    var CHART_DEFAULTS = {
        font:   { family: "'JetBrains Mono', monospace", size: 11 },
        color:  '#e0f0ff',
        muted:  '#7fa7c0',
        cyan:   'rgba(0, 180, 216, 0.85)',
        cyanBg: 'rgba(0, 180, 216, 0.12)',
        green:  'rgba(0, 200, 100, 0.75)',
        greenBg:'rgba(0, 200, 100, 0.08)',
        grid:   'rgba(255, 255, 255, 0.05)',
    };

    function renderBarChart(el, s) {
        withChart(function () {
            var canvas = el.querySelector('.rca-bar-canvas');
            if (!canvas) return;
            if (el._rcaBarChart) { el._rcaBarChart.destroy(); el._rcaBarChart = null; }
            var d = derive(s);
            var labels      = ['Embed Corpus', 'Embed Query', 'Reranker', 'Vector DB', 'LLM Input', 'LLM Output'];
            var ragData     = [d.embCorpus, d.embQuery, d.rerankCost, d.vdbCost, d.ragLLMInpCost, d.ragLLMOutCost];
            var cacheInpCost = s.cacheEnabled ? (d.cacheWrite + d.cacheRead + d.freshInp) : (d.cacheTotal - d.outCost);
            var cacheData   = [0, 0, 0, 0, cacheInpCost, d.outCost];
            var C = CHART_DEFAULTS;
            el._rcaBarChart = new window.Chart(canvas, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [
                        { label: 'RAG',       data: ragData,   backgroundColor: C.cyanBg,  borderColor: C.cyan,  borderWidth: 1 },
                        { label: 'Cache-Aug', data: cacheData, backgroundColor: C.greenBg, borderColor: C.green, borderWidth: 1 },
                    ],
                },
                options: {
                    indexAxis: 'y',
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { labels: { color: C.color, font: C.font } },
                        tooltip: { callbacks: { label: function (ctx) { return ' ' + ctx.dataset.label + ': ' + fmt(ctx.parsed.x); } } },
                    },
                    scales: {
                        x: { ticks: { color: C.muted, callback: function (v) { return fmt(v); } }, grid: { color: C.grid } },
                        y: { ticks: { color: C.muted, font: { family: C.font.family, size: 10 } }, grid: { color: C.grid } },
                    },
                },
            });
        });
    }

    function renderScaleChart(el, s) {
        var POINTS = [1, 10, 25, 50, 100, 250, 500, 1000];
        withChart(function () {
            var canvas = el.querySelector('.rca-scale-canvas');
            if (!canvas) return;
            if (el._rcaScaleChart) { el._rcaScaleChart.destroy(); el._rcaScaleChart = null; }
            var ragVals   = POINTS.map(function (n) { return derive(Object.assign({}, s, { nCalls: n })).ragTotal; });
            var cacheVals = POINTS.map(function (n) { return derive(Object.assign({}, s, { nCalls: n })).cacheTotal; });
            var C = CHART_DEFAULTS;
            el._rcaScaleChart = new window.Chart(canvas, {
                type: 'line',
                data: {
                    labels: POINTS.map(String),
                    datasets: [
                        { label: 'RAG',       data: ragVals,   borderColor: C.cyan,  backgroundColor: C.cyanBg,  fill: true, tension: 0.3, pointBackgroundColor: C.cyan },
                        { label: 'Cache-Aug', data: cacheVals, borderColor: C.green, backgroundColor: C.greenBg, fill: true, tension: 0.3, pointBackgroundColor: C.green },
                    ],
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { labels: { color: C.color, font: C.font } },
                        tooltip: { callbacks: { label: function (ctx) { return ' ' + ctx.dataset.label + ': ' + fmt(ctx.parsed.y); } } },
                    },
                    scales: {
                        x: { title: { display: true, text: 'LLM Calls', color: C.muted }, ticks: { color: C.muted }, grid: { color: C.grid } },
                        y: { title: { display: true, text: 'Total Cost', color: C.muted }, ticks: { color: C.muted, callback: function (v) { return fmt(v); } }, grid: { color: C.grid } },
                    },
                },
            });
        });
    }

    // =========================================================
    // WIDGET MOUNT + EVENT BINDING
    // =========================================================
    function mountWidget(placeholder) {
        if (placeholder._rcaMounted) return;
        placeholder._rcaMounted = true;

        var state = defaultState();

        function render() {
            // Destroy existing chart instances before replacing DOM
            if (placeholder._rcaBarChart)   { placeholder._rcaBarChart.destroy();   placeholder._rcaBarChart   = null; }
            if (placeholder._rcaScaleChart) { placeholder._rcaScaleChart.destroy(); placeholder._rcaScaleChart = null; }

            placeholder.innerHTML = buildWidget(state);
            bindEvents(placeholder, state, render);

            if (state.activeTab === 1) renderBarChart(placeholder, state);
            if (state.activeTab === 2) renderScaleChart(placeholder, state);
        }

        render();
    }

    function bindEvents(el, state, rerender) {
        // Selects
        el.querySelectorAll('select[data-bind]').forEach(function (sel) {
            sel.addEventListener('change', function () {
                var key = this.getAttribute('data-bind');
                state[key] = this.value;
                rerender();
            });
        });

        // Range sliders — update label live, full rerender on release
        el.querySelectorAll('input[type="range"][data-bind]').forEach(function (inp) {
            inp.addEventListener('input', function () {
                var key = this.getAttribute('data-bind');
                state[key] = parseFloat(this.value);
                var valEl = el.querySelector('.rca-slider-val[data-for="' + key + '"]');
                if (valEl) valEl.textContent = fmtN(state[key]);
            });
            inp.addEventListener('change', function () {
                var key = this.getAttribute('data-bind');
                state[key] = parseFloat(this.value);
                if (key === 'nCalls') state.cacheWriteEvents = Math.min(state.cacheWriteEvents, state.nCalls);
                rerender();
            });
        });

        // Number inputs
        el.querySelectorAll('input[type="number"][data-bind]').forEach(function (inp) {
            inp.addEventListener('change', function () {
                var key = this.getAttribute('data-bind');
                var v = parseFloat(this.value);
                if (isNaN(v)) return;
                state[key] = v;
                if (key === 'nCalls') state.cacheWriteEvents = Math.min(state.cacheWriteEvents, v);
                rerender();
            });
        });

        // Toggle button
        var toggleBtn = el.querySelector('.rca-toggle[data-bind="cacheEnabled"]');
        if (toggleBtn) {
            toggleBtn.addEventListener('click', function () {
                state.cacheEnabled = !state.cacheEnabled;
                rerender();
            });
        }

        // Tab buttons
        el.querySelectorAll('.rca-tab-btn').forEach(function (btn) {
            btn.addEventListener('click', function () {
                state.activeTab = parseInt(this.getAttribute('data-tab'), 10);
                rerender();
            });
        });
    }

    // =========================================================
    // PUBLIC API
    // =========================================================
    window.initRagCacheWidgets = function () {
        document.querySelectorAll('.rag-cache-widget-placeholder').forEach(mountWidget);
    };

}());
