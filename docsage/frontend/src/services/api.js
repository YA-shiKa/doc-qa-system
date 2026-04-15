// // frontend/src/services/api.js
// const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000/api/v1";

// async function request(path, options = {}) {
//   const res = await fetch(`${BASE_URL}${path}`, {
//     headers: { "Content-Type": "application/json", ...options.headers },
//     ...options,
//   });
//   if (!res.ok) {
//     const err = await res.json().catch(() => ({ detail: res.statusText }));
//     throw new Error(err.detail || "Request failed");
//   }
//   return res.json();
// }

// // ── Sessions ─────────────────────────────────────────────────────────────────
// export const createSession = (docIds = []) =>
//   request("/sessions/", {
//     method: "POST",
//     body: JSON.stringify({ doc_ids: docIds }),
//   });

// export const getHistory = (sessionId) =>
//   request(`/sessions/${sessionId}/history`);

// export const clearHistory = (sessionId) =>
//   request(`/sessions/${sessionId}/history`, { method: "DELETE" });

// // ── Documents ─────────────────────────────────────────────────────────────────
// export const uploadDocument = async (file, onProgress) => {
//   const formData = new FormData();
//   formData.append("file", file);

//   // Use XHR for upload progress tracking
//   return new Promise((resolve, reject) => {
//     const xhr = new XMLHttpRequest();
//     xhr.open("POST", `${BASE_URL}/documents/upload`);
//     xhr.upload.addEventListener("progress", (e) => {
//       if (e.lengthComputable && onProgress) {
//         onProgress(Math.round((e.loaded / e.total) * 100));
//       }
//     });
//     xhr.addEventListener("load", () => {
//       if (xhr.status >= 200 && xhr.status < 300) {
//         resolve(JSON.parse(xhr.responseText));
//       } else {
//         const err = JSON.parse(xhr.responseText || "{}");
//         reject(new Error(err.detail || "Upload failed"));
//       }
//     });
//     xhr.addEventListener("error", () => reject(new Error("Network error")));
//     xhr.send(formData);
//   });
// };

// export const pollDocument = (docId) => request(`/documents/${docId}`);

// export const listDocuments = () => request("/documents/");

// export const deleteDocument = (docId) =>
//   request(`/documents/${docId}`, { method: "DELETE" });

// // ── QA ────────────────────────────────────────────────────────────────────────
// export const askQuestion = (question, sessionId, docIds = null) =>
//   request("/qa/ask", {
//     method: "POST",
//     body: JSON.stringify({
//       question,
//       session_id: sessionId,
//       doc_ids: docIds,
//     }),
//   });
// frontend/src/services/api.js
//
// Render + Vercel deployment fixes:
//  1. WAKE_UP_TIMEOUT: Render free tier sleeps after 15min inactivity.
//     First request can take 30-60s. We use a long timeout + retry with
//     a "waking up server..." message so the user knows what's happening.
//  2. XHR for uploads: kept as-is (needed for progress events).
//  3. All timeouts configurable via constants at the top.

const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000/api/v1";

// How long to wait for the server to wake up (Render cold start)
const WAKE_TIMEOUT_MS = 70_000;   // 70 seconds
// Timeout for regular requests (QA can be slow on CPU)
const REQUEST_TIMEOUT_MS = 120_000; // 2 minutes
// How many times to retry on network failure
const MAX_RETRIES = 2;
const RETRY_DELAY_MS = 3000;

// ── Core fetch with timeout + retry ──────────────────────────────────────────

async function request(path, options = {}, timeoutMs = REQUEST_TIMEOUT_MS) {
  const url = `${BASE_URL}${path}`;
  let lastError;

  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    if (attempt > 0) {
      await sleep(RETRY_DELAY_MS);
    }

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);

    try {
      const res = await fetch(url, {
        headers: { "Content-Type": "application/json", ...options.headers },
        signal: controller.signal,
        ...options,
      });
      clearTimeout(timer);

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}: ${res.statusText}` }));
        throw new Error(err.detail || `Request failed with status ${res.status}`);
      }
      return await res.json();

    } catch (err) {
      clearTimeout(timer);
      lastError = err;

      // Don't retry on HTTP errors (4xx/5xx) — only on network failures
      if (err.name !== "AbortError" && err.message?.includes("HTTP ")) {
        throw err;
      }

      // AbortError = timeout, TypeError = network failure (e.g. ERR_CONNECTION_TIMED_OUT)
      const isNetworkError = err.name === "AbortError" || err instanceof TypeError;
      if (!isNetworkError || attempt === MAX_RETRIES) {
        break;
      }
      console.warn(`[DocSage] Request failed (attempt ${attempt + 1}), retrying...`, err.message);
    }
  }

  // Format a useful error message
  if (lastError?.name === "AbortError") {
    throw new Error("Request timed out. The server may be waking up — please try again in a moment.");
  }
  if (lastError instanceof TypeError) {
    throw new Error("Cannot reach the server. Check that the backend URL is correct and the service is running.");
  }
  throw lastError;
}

// ── Wake-up ping ──────────────────────────────────────────────────────────────
// Call this on app startup to wake Render before the user does anything.
// Returns true if server is up, false if it timed out.
export async function pingServer() {
  try {
    await request("/health", {}, WAKE_TIMEOUT_MS);
    return true;
  } catch (_) {
    return false;
  }
}

// ── Sessions ─────────────────────────────────────────────────────────────────
export const createSession = (docIds = []) =>
  request("/sessions/", {
    method: "POST",
    body: JSON.stringify({ doc_ids: docIds }),
  }, WAKE_TIMEOUT_MS); // Use long timeout — this is the first request after cold start

export const getHistory = (sessionId) =>
  request(`/sessions/${sessionId}/history`);

export const clearHistory = (sessionId) =>
  request(`/sessions/${sessionId}/history`, { method: "DELETE" });

// ── Documents ─────────────────────────────────────────────────────────────────
export const uploadDocument = async (file, onProgress) => {
  const formData = new FormData();
  formData.append("file", file);

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${BASE_URL}/documents/upload`);

    // Timeout for upload: 5 minutes (large PDFs + slow connections)
    xhr.timeout = 300_000;

    xhr.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable && onProgress) {
        onProgress(Math.round((e.loaded / e.total) * 100));
      }
    });
    xhr.addEventListener("load", () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText));
        } catch (_) {
          reject(new Error("Invalid response from server"));
        }
      } else {
        try {
          const err = JSON.parse(xhr.responseText || "{}");
          reject(new Error(err.detail || `Upload failed with status ${xhr.status}`));
        } catch (_) {
          reject(new Error(`Upload failed with status ${xhr.status}`));
        }
      }
    });
    xhr.addEventListener("error", () =>
      reject(new Error("Upload failed: network error. Check your connection."))
    );
    xhr.addEventListener("timeout", () =>
      reject(new Error("Upload timed out. Try a smaller file or check your connection."))
    );
    xhr.send(formData);
  });
};

export const pollDocument = (docId) => request(`/documents/${docId}`);
export const listDocuments = () => request("/documents/");
export const deleteDocument = (docId) =>
  request(`/documents/${docId}`, { method: "DELETE" });

// ── QA ────────────────────────────────────────────────────────────────────────
export const askQuestion = (question, sessionId, docIds = null) =>
  request("/qa/ask", {
    method: "POST",
    body: JSON.stringify({ question, session_id: sessionId, doc_ids: docIds }),
  }, REQUEST_TIMEOUT_MS);

// ── Utility ───────────────────────────────────────────────────────────────────
function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}
