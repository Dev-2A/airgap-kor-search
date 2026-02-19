/**
 * 에어갭 한국어 문서 검색 - 프론트엔드
 */

const API = {
    search: "/api/search",
    indexText: "/api/index/text",
    indexFile: "/api/index/file",
    documents: "/api/documents",
    stats: "/api/stats",
    health: "/api/health",
};

// ── 탭 전환 ──────────────────────────────────────────────

document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
        document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
        document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));

        tab.classList.add("active");
        const target = document.getElementById(`tab-${tab.dataset.tab}`);
        if (target) target.classList.add("active");

        // 관리 탭 진입 시 자동 새로고침
        if (tab.dataset.tab === "manage") {
            loadManageTab();
        }
    });
});

// ── 검색 ─────────────────────────────────────────────────

const searchInput = document.getElementById("search-input");
const searchBtn = document.getElementById("search-btn");
const searchResults = document.getElementById("search-results");
const searchMeta = document.getElementById("search-meta");
const searchEmpty = document.getElementById("search-empty");
const topKSelect = document.getElementById("top-k");

searchInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") doSearch();
});
searchBtn.addEventListener("click", doSearch);

async function doSearch() {
    const query = searchInput.value.trim();
    if (!query) return;

    searchBtn.disabled = true;
    searchResults.innerHTML = `<div class="loading"><span class="spinner"></span>검색 중...</div>`;
    searchMeta.style.display = "none";
    searchEmpty.style.display = "none";

    try {
        const res = await fetch(API.search, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                query,
                top_k: parseInt(topKSelect.value),
            }),
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "검색 실패");
        }

        const data = await res.json();
        renderSearchResults(data);
    } catch (err) {
        searchResults.innerHTML = `<div class="empty-state"><p>❌ ${err.message}</p></div>`;
    } finally {
        searchBtn.disabled = false;
    }
}

function renderSearchResults(data) {
    searchMeta.textContent = `"${data.query}" → ${data.total_found}건 (${data.elapsed_ms.toFixed(1)}ms)`;
    searchMeta.style.display = "block";

    if (data.results.length === 0) {
        searchResults.innerHTML = "";
        searchEmpty.style.display = "block";
        searchEmpty.innerHTML = `<p>😢 검색 결과가 없습니다.</p><p>다른 키워드로 시도해보세요.</p>`;
        return;
    }

    searchEmpty.style.display = "none";
    searchResults.innerHTML = data.results
        .map((r, i) => {
            const scoreClass =
                r.score >= 0.8 ? "score-high" : r.score >= 0.6 ? "score-mid" : "score-low";
            const preview = r.text.length > 300 ? r.text.slice(0, 300) + "..." : r.text;

            return `
                <div class="result-card">
                    <div class="result-header">
                        <span class="result-rank ${scoreClass}">#${i + 1} (${r.score_percent}%)</span>
                        <span class="result-source">${escapeHtml(r.doc_path)} · 청크 #${r.chunk_index}</span>
                    </div>
                    <div class="result-text">${escapeHtml(preview)}</div>
                </div>
            `;
        })
        .join("");
}

// ── 인덱싱: 텍스트 ──────────────────────────────────────

const indexTextBtn = document.getElementById("index-text-btn");
const textInput = document.getElementById("text-input");
const textSource = document.getElementById("text-source");
const indexResult = document.getElementById("index-result");

indexTextBtn.addEventListener("click", async () => {
    const text = textInput.value.trim();
    if (!text) return;

    indexTextBtn.disabled = true;
    showIndexResult("loading", "인덱싱 중...");

    try {
        const res = await fetch(API.indexText, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                text,
                source: textSource.value.trim() || "<직접 입력>",
            }),
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "인덱싱 실패");
        }

        const data = await res.json();
        showIndexResult(
            "success",
            `✅ 인덱싱 완료: ${data.chunks_created}개 청크 생성 (${data.elapsed_sec.toFixed(2)}초)`
        );
        textInput.value = "";
    } catch (err) {
        showIndexResult("error", `❌ ${err.message}`);
    } finally {
        indexTextBtn.disabled = false;
    }
});

// ── 인덱싱: 파일 ────────────────────────────────────────

const indexFileBtn = document.getElementById("index-file-btn");
const fileInput = document.getElementById("file-input");

indexFileBtn.addEventListener("click", async () => {
    const file = fileInput.files[0];
    if (!file) return;

    indexFileBtn.disabled = true;
    showIndexResult("loading", `"${file.name}" 인덱싱 중...`);

    try {
        const formData = new FormData();
        formData.append("file", file);

        const res = await fetch(API.indexFile, {
            method: "POST",
            body: formData,
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "파일 인덱싱 실패");
        }

        const data = await res.json();
        showIndexResult(
            "success",
            `✅ "${file.name}" 인덱싱 완료: ${data.chunks_created}개 청크 (${data.elapsed_sec.toFixed(2)}초)`
        );
        fileInput.value = "";
    } catch (err) {
        showIndexResult("error", `❌ ${err.message}`);
    } finally {
        indexFileBtn.disabled = false;
    }
});

function showIndexResult(type, message) {
    indexResult.style.display = "block";
    indexResult.className = `index-result ${type}`;

    if (type === "loading") {
        indexResult.innerHTML = `<span class="spinner"></span>${message}`;
    } else {
        indexResult.textContent = message;
    }
}

// ── 관리 탭 ──────────────────────────────────────────────

const refreshBtn = document.getElementById("refresh-btn");
const statsBox = document.getElementById("stats-box");
const docList = document.getElementById("doc-list");

refreshBtn.addEventListener("click", loadManageTab);

async function loadManageTab() {
    try {
        const [statsRes, docsRes] = await Promise.all([
            fetch(API.stats),
            fetch(API.documents),
        ]);

        const stats = await statsRes.json();
        const docs = await docsRes.json();

        renderStats(stats);
        renderDocList(docs.documents);
    } catch (err) {
        statsBox.innerHTML = `<p>❌ 데이터 로드 실패: ${err.message}</p>`;
    }
}

function renderStats(stats) {
    statsBox.innerHTML = `
        <div class="stat-card">
            <div class="stat-value">${stats.total_documents}</div>
            <div class="stat-label">문서</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">${stats.total_chunks}</div>
            <div class="stat-label">청크</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">${stats.total_vectors}</div>
            <div class="stat-label">벡터</div>
        </div>
    `;
}

function renderDocList(documents) {
    if (documents.length === 0) {
        docList.innerHTML = `<div class="empty-state"><p>인덱싱된 문서가 없습니다.</p></div>`;
        return;
    }

    docList.innerHTML = documents
        .map(
            (doc) => `
            <div class="doc-item">
                <div class="doc-info">
                    <div class="doc-path">${escapeHtml(doc.doc_path)}</div>
                    <div class="doc-chunks">${doc.chunk_count}개 청크</div>
                </div>
                <button class="btn btn-danger" onclick="deleteDoc('${escapeAttr(doc.doc_path)}')">삭제</button>
            </div>
        `
        )
        .join("");
}

async function deleteDoc(docPath) {
    if (!confirm(`"${docPath}"을(를) 삭제하시겠습니까?`)) return;

    try {
        const res = await fetch(`${API.documents}?doc_path=${encodeURIComponent(docPath)}`, {
            method: "DELETE",
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "삭제 실패");
        }

        loadManageTab();
    } catch (err) {
        alert(`삭제 실패: ${err.message}`);
    }
}

// ── 유틸리티 ─────────────────────────────────────────────

function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}

function escapeAttr(str) {
    return str.replace(/'/g, "\\'").replace(/"/g, '\\"');
}

// ── 초기 로드 ────────────────────────────────────────────

searchInput.focus();