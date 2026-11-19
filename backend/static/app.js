// 새 화면(첫 화면 home.html · 층 화면 level.html)이 함께 쓰는 상단 바, 사진 확대, 알림 카드.
// common.js 다음에 불러옵니다. 기존 화면(dashboard_classic.html)은 이 파일을 쓰지 않습니다.

// ===================== 상단 바 =====================
// active: "home" | "report" | "classic"
function renderTopBar(me, active) {
    const bar = document.getElementById("top-bar");
    if (!bar) return;
    const link = (key, href, label, icon) =>
        `<a href="${href}" class="px-3 py-2 rounded-lg whitespace-nowrap flex items-center gap-1.5 ${key === active
            ? "bg-indigo-50 text-indigo-700" : "text-slate-500 hover:bg-slate-100"}"><span class="md:hidden"><i class="fa-solid ${icon}"></i></span><span class="hidden md:inline">${label}</span></a>`;
    bar.className = "bg-white border-b border-slate-200 sticky top-0 z-30";
    bar.innerHTML = `
        <div class="max-w-6xl mx-auto px-4 md:px-8 h-16 flex items-center gap-3 md:gap-8">
            <a href="/home" class="flex items-center gap-2.5 shrink-0">
                <div class="w-9 h-9 rounded-xl bg-indigo-600 text-white flex items-center justify-center"><i class="fa-solid fa-book-open"></i></div>
                <div class="hidden sm:block"><div class="font-bold text-slate-900 leading-tight">서고 모니터링</div>
                    <div class="text-[11px] text-slate-400 leading-tight">${escapeHtml(me.library_name)}</div></div>
            </a>
            <nav class="flex gap-1 text-sm font-medium">
                ${link("home", "/home", "오늘의 서가", "fa-house")}
                ${link("report", "/report", "일일 리포트", "fa-file-lines")}
                ${link("classic", "/classic", me.is_admin ? "관리" : "전체 보기", "fa-gear")}
            </nav>
            <div class="ml-auto flex items-center gap-2 md:gap-3">
                <div class="text-right hidden sm:block"><div class="text-sm font-semibold text-slate-700">${escapeHtml(me.display_name)}</div>
                    <div class="text-[11px] text-slate-400">${me.is_admin ? "관리자" : "사서"}</div></div>
                <button onclick="logout()" title="로그아웃" class="w-9 h-9 rounded-full bg-slate-100 hover:bg-slate-200 text-slate-500 flex items-center justify-center">
                    <i class="fa-solid fa-right-from-bracket"></i></button>
            </div>
        </div>`;
}

// ===================== 사진 크게 보기 =====================
function openLightbox(url, title, caption = "") {
    let box = document.getElementById("lightbox");
    if (!box) {
        box = document.createElement("div");
        box.id = "lightbox";
        box.className = "fixed inset-0 z-50 hidden items-center justify-center bg-black/80 backdrop-blur-sm p-4";
        box.innerHTML = `
            <div class="relative max-w-4xl w-full" onclick="event.stopPropagation()">
                <button onclick="closeLightbox()" class="absolute -top-10 right-0 text-white/80 hover:text-white text-sm"><i class="fa-solid fa-xmark"></i> 닫기 (ESC)</button>
                <div class="bg-white rounded-2xl overflow-hidden">
                    <img id="lightbox-img" class="w-full max-h-[75vh] object-contain bg-slate-100" alt="">
                    <div class="px-4 py-3"><div id="lightbox-title" class="text-sm font-bold text-slate-700"></div>
                        <div id="lightbox-caption" class="text-xs text-slate-400 mt-0.5"></div></div>
                </div>
            </div>`;
        box.addEventListener("click", closeLightbox);
        document.body.appendChild(box);
        document.addEventListener("keydown", e => { if (e.key === "Escape") closeLightbox(); });
    }
    document.getElementById("lightbox-img").src = url;
    document.getElementById("lightbox-title").textContent = title;
    document.getElementById("lightbox-caption").textContent = caption;
    box.classList.remove("hidden");
    box.classList.add("flex");
}

function closeLightbox() {
    const box = document.getElementById("lightbox");
    if (box) { box.classList.add("hidden"); box.classList.remove("flex"); }
}

// 도서관 전체 지도 (관리 화면에서 올린 이미지). 있으면 true
function libraryMapUrl() {
    return `${API_BASE}/library-map?v=${Date.now()}`;
}

// ===================== 층 목록 =====================
// 첫 화면 목록 순서: 정리 필요 → 재촬영 → 확인만 필요 → 기준 사진 없음
const ATTENTION_ORDER = ["pending_action", "retake", "pending_check", "no_reference"];
const WORK_STATUSES = new Set(["pending_action", "retake"]);   // '정리가 필요한 곳'으로 세는 상태

// /api/locations 트리 → 층 목록 (구역 · 책꽂이 정보 포함)
function flattenLevels(tree) {
    const levels = [];
    tree.forEach(zone => zone.bookcases.forEach(bookcase => bookcase.levels.forEach(level =>
        levels.push({ ...level, zone, bookcase }))));
    return levels;
}

function attentionLevels(levels) {
    return levels.filter(l => ATTENTION_ORDER.includes(l.status))
        .sort((a, b) => (ATTENTION_ORDER.indexOf(a.status) - ATTENTION_ORDER.indexOf(b.status))
            || (b.latest_session?.summary.pending_count || 0) - (a.latest_session?.summary.pending_count || 0));
}

// "A구역 1번 책꽂이 5층" → 화면 제목용 { main: "1번 책꽂이 5층", zone: "A구역" }
function levelTitle(level) {
    return { main: `${level.bookcase.bookcase_no}번 책꽂이 ${level.level}층`, zone: `${level.zone.zone_code}구역` };
}

function levelUrl(level) {
    return `/level?shelf_id=${encodeURIComponent(level.shelf_id)}`;
}

// ===================== 알림 카드 =====================
// 오배가(다른 서가 책)만 펼쳐서 맨 위에, 나머지는 심각도 순으로 접어 둠
const OPEN_GROUPS = new Set(["foreign"]);
const DASHBOARD_GROUP_ORDER = [...ISSUE_GROUPS.filter(g => OPEN_GROUPS.has(g.key)), ...ISSUE_GROUPS.filter(g => !OPEN_GROUPS.has(g.key))];
const HANDLED_GROUP = { key: "handled", title: "처리된 알림", icon: "fa-circle-check", badge: "bg-emerald-100 text-emerald-700",
                        text: "text-emerald-700", hint: "처리 완료 · 오탐으로 표시한 알림" };

// 알림 라벨별 사서 안내 문구 (라벨/그룹 분류는 서버가 결정). 책은 제목으로 부르고 위치는 작게 덧붙임
const MESSAGE_BY_LABEL = {
    "누락/분실":     r => `사진과 RFID 모두에서 확인되지 않았습니다. 다른 책꽂이에 꽂혀 있는지 확인해 주세요.`,
    "오배가":        r => `이 서가의 책이 아닙니다. ${r.home_location
        ? `원래 위치 <b>${escapeHtml(r.home_location)}</b>로 옮겨 주세요.` : "배가 위치가 지정되지 않은 도서입니다. 도서 정보를 확인해 주세요."}`,
    "상태이상":      r => `대출 중(미비치)으로 되어 있는데 서가에서 발견되었습니다. 반납 처리를 확인해 주세요.`,
    "미분석":        r => `RFID로 읽은 등록번호의 도서 정보가 없습니다. 등록 여부를 확인해 주세요.`,
    "오배열":        r => `순서가 잘못되었습니다. ${r.move_to ? `<b>${escapeHtml(r.move_to)}</b>로 옮겨 주세요.` : "제자리로 옮겨 주세요."}`,
    "인식 실패":     r => `RFID는 감지되었으나 사진에서 찾지 못했습니다. 서가에서 직접 확인해 주세요.`,
    "미확인 도서":   r => `어떤 도서인지 식별하지 못했습니다. 직접 확인해 주세요.`,
    "중복 인식":     r => `다른 위치의 책과 같은 도서로 인식되었습니다. 직접 확인해 주세요.`,
    "RFID 미인식":   r => `사진에서는 확인되었으나 RFID 태그가 감지되지 않았습니다. 태그를 점검해 주세요.`,
    "뒤집힘":        r => `거꾸로 꽂혀 있습니다. 똑바로 세워 주세요.`,
    "기울어짐":      r => `기울어져 있습니다. 정돈해 주세요.`,
    "가로로 누움":   r => `가로로 눕혀져 있거나 다른 책 위에 쌓여 있습니다. 세워서 꽂아 주세요.`,
    "책등 안 보임": r => `책등이 보이지 않아 어떤 책인지 알 수 없습니다. 거꾸로 꽂혔거나 종이가 끼어 있을 수 있으니 확인해 주세요.`,
};

// 카드 제목: 「제목」 / 등록번호 / 왼쪽에서 N번째 책
function cardTitle(row) {
    if (row.title) return `「${escapeHtml(row.title)}」`;
    if (row.reg_no && row.reg_no !== "UNKNOWN") return `등록번호 ${escapeHtml(row.reg_no)}`;
    return row.sequence_order > 0 ? `왼쪽에서 ${row.sequence_order}번째 책` : "위치를 알 수 없는 책";
}

function iconButton(title, icon, classes, onClick, label = "") {
    const btn = document.createElement("button");
    btn.className = `h-9 ${label ? "px-3.5" : "w-9"} rounded-lg flex items-center justify-center gap-1.5 text-sm font-semibold transition-colors ${classes}`;
    btn.title = title;
    btn.setAttribute("aria-label", title);
    btn.innerHTML = `<i class="fa-solid ${icon}"></i>${label ? `<span>${label}</span>` : ""}`;
    btn.addEventListener("click", onClick);
    return btn;
}

// onAction(row, status): 처리 완료 / 오탐 / 되돌리기
function buildTaskCard(row, onAction) {
    const [primary, ...others] = row.issues;
    const group = GROUP_INFO[primary.group];
    const handled = row.action_status !== "PENDING";
    const message = (MESSAGE_BY_LABEL[primary.label] || (() => ""))(row);
    const seq = (row.title || (row.reg_no && row.reg_no !== "UNKNOWN")) && row.sequence_order > 0
        ? `<span class="text-xs font-normal text-slate-400 ml-1">왼쪽에서 ${row.sequence_order}번째</span>` : "";
    const extra = others.map(o => `<span class="ml-1 text-[11px] font-semibold px-1.5 py-0.5 rounded ${GROUP_INFO[o.group].badge}">+ ${escapeHtml(o.label)}</span>`).join("");
    const done = handled
        ? `<div class="text-[11px] font-semibold text-emerald-700 mt-1">${ACTION_LABELS[row.action_status]}${row.action_by_name ? " · " + escapeHtml(row.action_by_name) : ""} · ${formatTime(row.action_time)}</div>` : "";
    const spine = spineUrl(row.spine_img_path);

    const card = document.createElement("div");
    card.className = "rounded-xl border border-slate-200 border-l-4 p-3 flex gap-3 "
        + (handled ? "border-l-slate-300 bg-slate-50 text-slate-400" : `${group.stripe} bg-white text-slate-700`);
    card.innerHTML = `
        ${spine ? `<button type="button" data-spine class="w-12 h-24 sm:w-14 sm:h-28 shrink-0 rounded-lg bg-slate-100 overflow-hidden cursor-zoom-in ${handled ? "opacity-50" : ""}"
            style="background:url('${spine}') center/contain no-repeat, #f1f5f9" title="책등 사진 크게 보기"></button>` : ""}
        <div class="flex-1 min-w-0">
            <span class="text-[11px] font-bold px-1.5 py-0.5 rounded ${handled ? "bg-slate-200 text-slate-500" : group.badge}">${escapeHtml(primary.label)}</span>${extra}
            <div class="font-bold mt-1.5 break-words ${handled ? "line-through decoration-slate-300" : "text-slate-900 sm:text-lg"}">${cardTitle(row)}${seq}</div>
            <div class="text-sm mt-0.5 leading-snug break-words ${handled ? "" : "text-slate-500"}">${message}</div>
            ${done}
            <div class="flex gap-2 mt-2.5" data-actions></div>
        </div>`;
    if (spine) card.querySelector("[data-spine]").addEventListener("click",
        () => openLightbox(spine, `${cardTitle(row).replace(/<[^>]+>/g, "")}`, row.final_status || ""));
    const actions = card.querySelector("[data-actions]");
    if (handled) {
        actions.appendChild(iconButton("되돌리기", "fa-rotate-left", "bg-white border border-slate-200 text-slate-500 hover:bg-slate-100",
            () => onAction(row, "PENDING"), "되돌리기"));
    } else {
        actions.appendChild(iconButton("처리 완료", "fa-check", "bg-emerald-600 text-white hover:bg-emerald-700",
            () => onAction(row, "RESOLVED"), "처리 완료"));
        actions.appendChild(iconButton("오탐", "fa-ban", "bg-white border border-slate-200 text-slate-500 hover:bg-slate-100",
            () => onAction(row, "FALSE_POSITIVE"), "오탐"));
    }
    return card;
}

// 접힌 그룹 미리보기: 앞의 두 권 이름 + 나머지 권수
function sectionPreview(rows) {
    const names = rows.slice(0, 2).map(r => cardTitle(r));
    return names.join(", ") + (rows.length > 2 ? ` 외 ${rows.length - 2}권` : "");
}

// 알림 그룹: 접혀 있으면 한 줄(설명 · 미리보기), 펼치면 색 제목 아래 카드 목록
function buildSection(group, rows, open, onAction) {
    const section = document.createElement("details");
    section.className = "group/sec";
    section.dataset.group = group.key;
    section.open = open;
    section.innerHTML = `
        <summary class="list-none [&::-webkit-details-marker]:hidden cursor-pointer select-none flex items-center gap-2 px-3.5 py-3 rounded-xl text-sm
                        bg-white border border-slate-200 hover:bg-slate-50
                        group-open/sec:bg-transparent group-open/sec:border-transparent group-open/sec:px-0 group-open/sec:hover:bg-transparent">
            <span class="w-7 h-7 shrink-0 rounded-lg flex items-center justify-center ${group.badge}"><i class="fa-solid ${group.icon} text-xs"></i></span>
            <span class="font-bold whitespace-nowrap ${group.text}">${group.title}</span>
            <span class="text-xs font-bold px-2 py-0.5 rounded-full whitespace-nowrap ${group.badge}">${rows.length}권</span>
            <span class="flex-1 min-w-0 text-xs text-slate-500 truncate">${group.hint || ""}<span class="group-open/sec:hidden"> · ${sectionPreview(rows)}</span></span>
            <span class="text-xs font-semibold text-slate-500 whitespace-nowrap group-open/sec:hidden">펼쳐 보기 <i class="fa-solid fa-chevron-down"></i></span>
            <span class="hidden group-open/sec:inline text-xs text-slate-400"><i class="fa-solid fa-chevron-up"></i></span>
        </summary>
        <div class="flex flex-col gap-2.5 mt-1 mb-3"></div>`;
    const list = section.querySelector("div");
    rows.forEach(row => list.appendChild(buildTaskCard(row, onAction)));
    return section;
}

function noticeBox(classes, icon, html) {
    const box = document.createElement("div");
    box.className = `p-3.5 rounded-xl border-l-4 text-sm flex items-start gap-3 ${classes}`;
    box.innerHTML = `<i class="fa-solid ${icon} mt-0.5"></i><div class="flex-1 min-w-0 break-words">${html}</div>`;
    return box;
}
