// Лента аниме: история хранится в браузере, лента и поиск – запросы к API (serving.api).
const API = (window.API_URL || "http://localhost:8000").replace(/\/$/, "");
const STORAGE_KEY = "anime-feed-state";

// состояние: досмотренное и скрытое – {anime_id, title}
let state = loadState();

function loadState() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY));
    if (saved && Array.isArray(saved.history) && Array.isArray(saved.excluded)) return saved;
  } catch (error) {
    // нет доступа к хранилищу или испорченные данные – начинаем с пустой истории
  }
  return { history: [], excluded: [] };
}

function saveState() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch (error) {
    // хранилище недоступно: лента работает, но история не переживёт перезагрузку
  }
}

async function api(path, options = {}) {
  const response = await fetch(API + path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

function plural(n, one, few, many) {
  const mod10 = n % 10;
  const mod100 = n % 100;
  if (mod10 === 1 && mod100 !== 11) return one;
  if (mod10 >= 2 && mod10 <= 4 && (mod100 < 12 || mod100 > 14)) return few;
  return many;
}

// ---------- лента ----------

let feedRequest = 0;

async function loadFeed() {
  const requestId = ++feedRequest;
  const status = document.getElementById("status");
  const list = document.getElementById("feed");
  list.setAttribute("aria-busy", "true");
  if (!list.children.length) status.textContent = "Подбираем…";
  try {
    const body = {
      history: state.history.map((item) => item.anime_id),
      exclude: state.excluded.map((item) => item.anime_id),
    };
    const feed = await api("/feed", { method: "POST", body: JSON.stringify(body) });
    if (requestId !== feedRequest) return; // пришёл ответ на устаревший запрос
    renderFeed(feed);
    status.textContent = "";
  } catch (error) {
    if (requestId !== feedRequest) return;
    status.textContent = `Не удалось получить ленту (${error.message}). Запущен ли API на ${API}?`;
  } finally {
    if (requestId === feedRequest) list.removeAttribute("aria-busy");
  }
}

const KINDS = { tv: "Сериал", movie: "Фильм", ova: "OVA", ona: "ONA", special: "Спешл", tv_special: "Спешл" };

function describe(anime) {
  const parts = [];
  if (anime.kind) parts.push(KINDS[anime.kind] || anime.kind);
  if (anime.year) parts.push(anime.year);
  if (anime.episodes && anime.episodes > 1) parts.push(`${anime.episodes} эп.`);
  if (anime.genres) parts.push(anime.genres.split(", ").slice(0, 2).join(", "));
  return parts.join(" · ");
}

function renderFeed(feed) {
  const personal = feed.strategy === "personal";
  document.getElementById("feed-title").textContent = personal ? "Ваша лента" : "Популярное";
  const badge = document.getElementById("feed-strategy");
  badge.hidden = false;
  badge.textContent = personal
    ? `по ${feed.history_used} ${plural(feed.history_used, "досмотренному", "досмотренным", "досмотренным")}`
    : "пока без истории";

  const list = document.getElementById("feed");
  const template = document.getElementById("card-template");
  list.replaceChildren();
  for (const item of feed.items) {
    const anime = item.anime;
    const card = template.content.firstElementChild.cloneNode(true);
    const poster = card.querySelector(".feed-card__poster");
    poster.href = anime.url;
    poster.setAttribute("aria-label", `${anime.title} на Shikimori`);
    card.querySelector(".feed-card__fallback").textContent = anime.title;
    const image = card.querySelector("img");
    // нет постера или он не загрузился – показываем название на фоне
    // в карточке постер крупнее превью (превью 225 px растягивалось бы)
    const posterUrl = anime.poster_main_url || anime.poster_url;
    if (posterUrl) {
      image.src = posterUrl;
      image.addEventListener("error", () => poster.classList.add("is-missing"));
    } else {
      poster.classList.add("is-missing");
    }
    card.querySelector(".feed-card__title").textContent = anime.title;
    card.querySelector(".feed-card__meta").textContent = describe(anime);
    card.querySelector(".feed-card__reason").textContent = item.reason;
    card.querySelector(".watched").addEventListener("click", () => addTo("history", anime));
    card.querySelector(".skip-item").addEventListener("click", () => addTo("excluded", anime));
    list.append(card);
  }
}

// ---------- история и скрытое ----------

function addTo(listName, anime) {
  const entry = { anime_id: anime.anime_id, title: anime.title };
  // аниме в одном списке: досмотренное не может быть одновременно скрытым
  state.history = state.history.filter((item) => item.anime_id !== entry.anime_id);
  state.excluded = state.excluded.filter((item) => item.anime_id !== entry.anime_id);
  state[listName].push(entry);
  saveState();
  renderLists();
  loadFeed();
}

function removeFrom(listName, animeId) {
  state[listName] = state[listName].filter((item) => item.anime_id !== animeId);
  saveState();
  renderLists();
  loadFeed();
}

function renderList(listName, elementId, emptyId) {
  const list = document.getElementById(elementId);
  list.replaceChildren();
  for (const item of state[listName]) {
    const chip = document.createElement("li");
    chip.className = "pill";
    const label = document.createElement("span");
    label.textContent = item.title;
    label.title = item.title;
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "chip-remove";
    remove.textContent = "×";
    remove.setAttribute("aria-label", `Убрать «${item.title}»`);
    remove.addEventListener("click", () => removeFrom(listName, item.anime_id));
    chip.append(label, remove);
    list.append(chip);
  }
  document.getElementById(emptyId).hidden = state[listName].length > 0;
}

function renderLists() {
  renderList("history", "history", "history-empty");
  renderList("excluded", "excluded", "excluded-empty");
}

// ---------- поиск ----------

let searchTimer = null;
let searchRequest = 0;
let active = -1;

function closeResults() {
  const input = document.getElementById("search");
  document.getElementById("results").hidden = true;
  input.setAttribute("aria-expanded", "false");
  input.removeAttribute("aria-activedescendant");
  active = -1;
}

function highlight(index) {
  const options = [...document.querySelectorAll("#results li[role='option']")];
  if (!options.length) return;
  active = (index + options.length) % options.length;
  options.forEach((option, i) => option.setAttribute("aria-selected", String(i === active)));
  options[active].scrollIntoView({ block: "nearest" });
  document.getElementById("search").setAttribute("aria-activedescendant", options[active].id);
}

function setupSearch() {
  const input = document.getElementById("search");

  input.addEventListener("input", () => {
    clearTimeout(searchTimer);
    const query = input.value.trim();
    if (!query) {
      closeResults();
      return;
    }
    // ждём паузу в наборе, чтобы не слать запрос на каждую букву
    searchTimer = setTimeout(() => runSearch(query), 250);
  });

  input.addEventListener("keydown", (event) => {
    const results = document.getElementById("results");
    if (event.key === "Escape") closeResults();
    if (results.hidden) return;
    if (event.key === "ArrowDown") { event.preventDefault(); highlight(active + 1); }
    if (event.key === "ArrowUp") { event.preventDefault(); highlight(active - 1); }
    if (event.key === "Enter") {
      event.preventDefault();
      const options = results.querySelectorAll("li[role='option']");
      const chosen = options[Math.max(active, 0)];
      if (chosen) chosen.click();
    }
  });

  document.addEventListener("click", (event) => {
    if (!event.target.closest(".search")) closeResults();
  });
}

async function runSearch(query) {
  const requestId = ++searchRequest;
  const input = document.getElementById("search");
  const results = document.getElementById("results");
  try {
    const found = await api(`/anime/search?q=${encodeURIComponent(query)}&limit=10`);
    if (requestId !== searchRequest) return;
    results.replaceChildren();
    active = -1;
    if (!found.length) {
      const empty = document.createElement("li");
      empty.className = "is-empty";
      empty.textContent = "В каталоге такого нет";
      results.append(empty);
    }
    found.forEach((anime, index) => {
      const row = document.createElement("li");
      row.id = `result-${index}`;
      row.setAttribute("role", "option");
      row.setAttribute("aria-selected", "false");
      let picture;
      if (anime.poster_url) {
        picture = document.createElement("img");
        picture.alt = "";
        picture.referrerPolicy = "no-referrer";
        picture.src = anime.poster_url;
      } else {
        picture = document.createElement("span");
        picture.className = "results__ph";
      }
      const text = document.createElement("div");
      const name = document.createElement("span");
      name.textContent = anime.title;
      const meta = document.createElement("small");
      meta.textContent = [anime.name !== anime.title ? anime.name : null, anime.year].filter(Boolean).join(" · ");
      text.append(name, meta);
      row.append(picture, text);
      row.addEventListener("click", () => {
        addTo("history", anime);
        input.value = "";
        closeResults();
        input.focus();
      });
      results.append(row);
    });
    results.hidden = false;
    input.setAttribute("aria-expanded", "true");
  } catch (error) {
    if (requestId !== searchRequest) return;
    results.replaceChildren();
    const failed = document.createElement("li");
    failed.className = "is-empty";
    failed.textContent = `Поиск недоступен (${error.message})`;
    results.append(failed);
    results.hidden = false;
  }
}

// ---------- старт ----------

document.getElementById("reset").addEventListener("click", () => {
  state = { history: [], excluded: [] };
  saveState();
  renderLists();
  loadFeed();
});

// ?history=1,30,47 – открыть ленту с этой историей (ссылка для демо); иначе – сохранённая в браузере
async function historyFromUrl() {
  const raw = new URLSearchParams(location.search).get("history");
  if (!raw) return;
  const ids = raw.split(",").map((part) => Number(part.trim())).filter(Number.isInteger).slice(0, 50);
  const cards = await Promise.all(ids.map((id) => api(`/anime/${id}`).catch(() => null)));
  const known = cards.filter(Boolean);
  if (!known.length) return;
  state = { history: known.map((anime) => ({ anime_id: anime.anime_id, title: anime.title })), excluded: [] };
  saveState();
}

async function start() {
  setupSearch();
  try {
    await historyFromUrl();
  } catch (error) {
    // API недоступен: покажем сохранённое, а loadFeed напишет, что не так
  }
  renderLists();
  loadFeed();
}

start();
