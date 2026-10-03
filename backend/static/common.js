// 대시보드(dashboard.html)와 일일 리포트(report.html)가 함께 쓰는 설정과 유틸리티
const API_BASE = "/api";
const SPINE_BASE = "/api/files/spine";   // 세션 원본 사진 · 책등 크롭 (로그인한 도서관 것만 내려줌)

// API 호출: 로그인이 만료되었으면(401) 로그인 화면으로 보냄. 실패하면 서버의 detail 메시지로 에러
async function fetchJson(url, options) {
    const res = await fetch(url, { credentials: "same-origin", ...options });
    if (res.status === 401) {
        location.href = "/login";
        throw new Error("로그인이 필요합니다.");
    }
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body.detail || `요청 실패 (HTTP ${res.status})`);
    return body;
}

// 로그인한 사서 정보 { library_id, library_name, display_name, role, is_admin, ... }
async function loadMe() {
    return fetchJson(`${API_BASE}/auth/me`);
}

async function logout() {
    await fetch(`${API_BASE}/auth/logout`, { method: "POST", credentials: "same-origin" });
    location.href = "/login";
}

// 상단 바의 도서관 · 사서 표시 + 로그아웃 버튼 (요소 id: user-box)
function renderUserBox(me) {
    const box = document.getElementById("user-box");
    if (!box) return;
    const role = me.is_admin
        ? `<span class="text-[10px] font-bold px-1.5 py-0.5 rounded bg-indigo-100 text-indigo-700">관리자</span>`
        : `<span class="text-[10px] font-bold px-1.5 py-0.5 rounded bg-slate-100 text-slate-600">사서</span>`;
    box.innerHTML = `
        <span class="text-sm text-slate-600 flex items-center gap-1.5">
            <i class="fa-solid fa-building-columns text-slate-400"></i>
            <b class="text-slate-700">${escapeHtml(me.library_name)}</b>
            <span class="text-slate-300">|</span> ${escapeHtml(me.display_name)} ${role}
        </span>
        <button onclick="logout()" class="text-xs text-slate-500 hover:text-slate-800 border border-slate-200 rounded-lg px-2 py-1">
            <i class="fa-solid fa-right-from-bracket"></i> 로그아웃
        </button>`;
}

// 알림 그룹 표시 설정. 순서 = 심각도 순서이며, 그룹 분류 자체는 서버(analyzer.classify_issues)가 결정합니다.
const ISSUE_GROUPS = [
    { key: "lost",     title: "분실 위험", icon: "fa-eye-slash",            color: "border-red-500 bg-red-50 text-red-700",          badge: "bg-red-100 text-red-700" },
    { key: "foreign",  title: "오배가",    icon: "fa-right-left",           color: "border-orange-500 bg-orange-50 text-orange-700", badge: "bg-orange-100 text-orange-700" },
    { key: "misorder", title: "오배열",    icon: "fa-shuffle",              color: "border-violet-500 bg-violet-50 text-violet-700", badge: "bg-violet-100 text-violet-700" },
    { key: "physical", title: "외형 이상", icon: "fa-triangle-exclamation", color: "border-amber-500 bg-amber-50 text-amber-700",    badge: "bg-amber-100 text-amber-700" },
    { key: "check",    title: "확인 필요", icon: "fa-magnifying-glass",     color: "border-slate-400 bg-slate-50 text-slate-700",    badge: "bg-slate-200 text-slate-700" },
];
const GROUP_INFO = Object.fromEntries(ISSUE_GROUPS.map(g => [g.key, g]));
const GROUP_RANK = Object.fromEntries(ISSUE_GROUPS.map((g, i) => [g.key, i]));

const ACTION_LABELS = { RESOLVED: "처리 완료", FALSE_POSITIVE: "오탐" };

// 서가 현황 맵의 층(칸) 상태 표시 (상태 판정은 서버 main.level_status)
const LEVEL_STATUS = {
    pending_action: { text: "조치 필요", cell: "bg-rose-100 border-rose-400 text-rose-800",        dot: "bg-rose-500" },
    retake:         { text: "재촬영",    cell: "bg-orange-100 border-orange-400 text-orange-800",  dot: "bg-orange-500" },
    pending_check:  { text: "확인 필요", cell: "bg-amber-50 border-amber-300 text-amber-800",      dot: "bg-amber-400" },
    ok:             { text: "정상",      cell: "bg-emerald-50 border-emerald-300 text-emerald-800", dot: "bg-emerald-500" },
    no_reference:   { text: "기준 사진 없음", cell: "bg-violet-50 border-violet-300 border-dashed text-violet-700", dot: "bg-violet-400" },
    unpatrolled:    { text: "미순찰",    cell: "bg-white border-slate-300 border-dashed text-slate-500", dot: "bg-white border border-slate-400" },
    no_books:       { text: "도서 없음", cell: "bg-slate-100 border-slate-200 text-slate-300",     dot: "bg-slate-200" },
};

// "A구역 1번 책꽂이 3층" (서버가 내려준 location_label이 없으면 shelf_id)
function locationText(item) {
    return escapeHtml(item.location_label || item.shelf_id || "");
}

function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, ch => (
        { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]
    ));
}

// "2026-10-03T02:30:58" → "10-03 02:30" (withDate=false면 "02:30")
function formatTime(value, withDate = true) {
    if (!value) return "-";
    const [d, t] = String(value).replace(" ", "T").split("T");
    const hm = (t || "").slice(0, 5);
    return withDate ? `${d.slice(5)} ${hm}` : hm;
}

function seqLabel(row) {
    return row.sequence_order > 0 ? `${row.sequence_order}번째` : "위치 불명";
}

// 도서 ID + 서명 (서가 정보 BOOK_MASTER에 등록된 도서만 서명이 있음)
function bookLabel(row) {
    const id = escapeHtml(row.book_id || "Unknown");
    return row.title ? `${id} 「${escapeHtml(row.title)}」` : id;
}

function spineUrl(path) {
    return path ? `${SPINE_BASE}/${path.split("/").map(encodeURIComponent).join("/")}` : "";
}

// 알림이 있고(issues) 아직 처리하지 않은(PENDING) 결과인지
function isPending(row) {
    return row.issues.length > 0 && row.action_status === "PENDING";
}

function sortByPriority(rows) {
    return [...rows].sort((a, b) => (GROUP_RANK[a.issues[0].group] - GROUP_RANK[b.issues[0].group])
                                 || (a.sequence_order - b.sequence_order));
}
