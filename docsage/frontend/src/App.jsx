// // frontend/src/App.jsx
// import { useState, useEffect } from "react";
// import DocumentUploader from "./components/DocumentUploader";
// import ChatInterface from "./components/ChatInterface";
// import DocumentLibrary from "./components/DocumentLibrary";
// import SessionBadge from "./components/SessionBadge";
// import { createSession } from "./services/api";
// import "./styles/globals.css";

// export default function App() {
//   const [sessionId, setSessionId] = useState(null);
//   const [documents, setDocuments] = useState([]);
//   const [activeDocIds, setActiveDocIds] = useState([]);
//   const [view, setView] = useState("chat"); // "chat" | "library"
//   const [isCreatingSession, setIsCreatingSession] = useState(true);

//   useEffect(() => {
//     const initSession = async () => {
//       try {
//         const session = await createSession([]);
//         setSessionId(session.session_id);
//       } catch (e) {
//         console.error("Session creation failed:", e);
//         // Use a local fallback session ID
//         setSessionId(`local_${Date.now()}`);
//       } finally {
//         setIsCreatingSession(false);
//       }
//     };
//     initSession();
//   }, []);

//   const handleDocumentReady = (doc) => {
//     setDocuments((prev) => {
//       const exists = prev.find((d) => d.doc_id === doc.doc_id);
//       if (exists) return prev.map((d) => (d.doc_id === doc.doc_id ? doc : d));
//       return [...prev, doc];
//     });
//   };

//   const handleDocumentDelete = (docId) => {
//     setDocuments((prev) => prev.filter((d) => d.doc_id !== docId));
//     setActiveDocIds((prev) => prev.filter((id) => id !== docId));
//   };

//   const toggleDocActive = (docId) => {
//     setActiveDocIds((prev) =>
//       prev.includes(docId) ? prev.filter((id) => id !== docId) : [...prev, docId]
//     );
//   };

//   if (isCreatingSession) {
//     return (
//       <div className="loading-screen">
//         <div className="loading-spinner" />
//         <p>Initializing DocSage…</p>
//       </div>
//     );
//   }

//   return (
//     <div className="app-shell">
//       {/* Sidebar */}
//       <aside className="sidebar">
//         <div className="sidebar-header">
//           <div className="logo">
//             <span className="logo-icon">◈</span>
//             <span className="logo-text">DocSage</span>
//           </div>
//           <p className="logo-tagline">Smart Document QA</p>
//         </div>

//         <nav className="sidebar-nav">
//           <button
//             className={`nav-item ${view === "chat" ? "active" : ""}`}
//             onClick={() => setView("chat")}
//           >
//             <span className="nav-icon">⬡</span> Chat
//           </button>
//           <button
//             className={`nav-item ${view === "library" ? "active" : ""}`}
//             onClick={() => setView("library")}
//           >
//             <span className="nav-icon">⬢</span> Library
//             {documents.length > 0 && (
//               <span className="nav-badge">{documents.length}</span>
//             )}
//           </button>
//         </nav>

//         <div className="sidebar-upload">
//           <DocumentUploader onDocumentReady={handleDocumentReady} />
//         </div>

//         {/* Active filters */}
//         {documents.length > 0 && (
//           <div className="active-docs">
//             <p className="active-docs-label">Search scope</p>
//             {documents
//               .filter((d) => d.status === "ready")
//               .map((doc) => (
//                 <button
//                   key={doc.doc_id}
//                   className={`doc-filter-chip ${
//                     activeDocIds.includes(doc.doc_id) ? "selected" : ""
//                   }`}
//                   onClick={() => toggleDocActive(doc.doc_id)}
//                   title={doc.filename}
//                 >
//                   <span className="chip-dot" />
//                   {doc.filename.length > 24
//                     ? doc.filename.slice(0, 22) + "…"
//                     : doc.filename}
//                 </button>
//               ))}
//             {activeDocIds.length > 0 && (
//               <button
//                 className="clear-filter"
//                 onClick={() => setActiveDocIds([])}
//               >
//                 Clear filter
//               </button>
//             )}
//           </div>
//         )}

//         <div className="sidebar-footer">
//           <SessionBadge sessionId={sessionId} />
//         </div>
//       </aside>

//       {/* Main content */}
//       <main className="main-content">
//         {view === "chat" ? (
//           <ChatInterface
//             sessionId={sessionId}
//             activeDocIds={activeDocIds.length > 0 ? activeDocIds : null}
//             hasDocuments={documents.some((d) => d.status === "ready")}
//           />
//         ) : (
//           <DocumentLibrary
//             documents={documents}
//             activeDocIds={activeDocIds}
//             onToggleActive={toggleDocActive}
//             onDelete={handleDocumentDelete}
//           />
//         )}
//       </main>
//     </div>
//   );
// }
// frontend/src/App.jsx
import { useState, useEffect } from "react";
import DocumentUploader from "./components/DocumentUploader";
import ChatInterface from "./components/ChatInterface";
import DocumentLibrary from "./components/DocumentLibrary";
import SessionBadge from "./components/SessionBadge";
import { createSession, pingServer } from "./services/api";
import "./styles/globals.css";

// How long to attempt waking the server before showing a hard error
const MAX_WAKE_ATTEMPTS = 3;

export default function App() {
  const [sessionId, setSessionId] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [activeDocIds, setActiveDocIds] = useState([]);
  const [view, setView] = useState("chat");

  // "booting" | "waking" | "ready" | "error"
  const [serverState, setServerState] = useState("booting");
  const [wakeAttempt, setWakeAttempt] = useState(0);
  const [errorMsg, setErrorMsg] = useState("");

  useEffect(() => {
    initApp();
  }, []);

  async function initApp() {
    setServerState("booting");

    // Step 1: Ping the server — handles Render cold-start wake-up
    const isUp = await tryWake();
    if (!isUp) return; // error state already set inside tryWake

    // Step 2: Create session
    try {
      const session = await createSession([]);
      setSessionId(session.session_id);
      setServerState("ready");
    } catch (e) {
      console.error("Session creation failed:", e);
      // Session creation failed but server is up — use local fallback
      // so the UI is still usable (session state is in-memory anyway)
      setSessionId(`local_${Date.now()}`);
      setServerState("ready");
    }
  }

  async function tryWake() {
    setServerState("waking");
    for (let i = 1; i <= MAX_WAKE_ATTEMPTS; i++) {
      setWakeAttempt(i);
      const ok = await pingServer();
      if (ok) return true;
      if (i < MAX_WAKE_ATTEMPTS) {
        // Brief pause between attempts
        await new Promise((r) => setTimeout(r, 2000));
      }
    }
    setServerState("error");
    setErrorMsg(
      `Could not reach the backend at ${import.meta.env.VITE_API_URL || "http://localhost:8000/api/v1"}. ` +
      "Check that VITE_API_URL is set correctly in Vercel environment variables and that the Render service is deployed."
    );
    return false;
  }

  const handleDocumentReady = (doc) => {
    setDocuments((prev) => {
      const exists = prev.find((d) => d.doc_id === doc.doc_id);
      if (exists) return prev.map((d) => (d.doc_id === doc.doc_id ? doc : d));
      return [...prev, doc];
    });
  };

  const handleDocumentDelete = (docId) => {
    setDocuments((prev) => prev.filter((d) => d.doc_id !== docId));
    setActiveDocIds((prev) => prev.filter((id) => id !== docId));
  };

  const toggleDocActive = (docId) => {
    setActiveDocIds((prev) =>
      prev.includes(docId) ? prev.filter((id) => id !== docId) : [...prev, docId]
    );
  };

  // ── Loading / wake-up screen ──────────────────────────────────────────────
  if (serverState === "booting" || serverState === "waking") {
    return (
      <div className="loading-screen">
        <div className="loading-spinner" />
        {serverState === "booting" && <p>Starting DocSage…</p>}
        {serverState === "waking" && (
          <div style={{ textAlign: "center" }}>
            <p>Waking up the server…</p>
            <p className="loading-sub">
              {wakeAttempt < MAX_WAKE_ATTEMPTS
                ? `Attempt ${wakeAttempt} of ${MAX_WAKE_ATTEMPTS} — this can take up to 60 seconds on first load`
                : "Almost there…"}
            </p>
          </div>
        )}
      </div>
    );
  }

  // ── Error screen ──────────────────────────────────────────────────────────
  if (serverState === "error") {
    return (
      <div className="loading-screen">
        <div className="error-icon">!</div>
        <p style={{ fontWeight: 600 }}>Cannot connect to backend</p>
        <p className="loading-sub" style={{ maxWidth: 480, textAlign: "center" }}>
          {errorMsg}
        </p>
        <div className="error-help">
          <p>Common fixes:</p>
          <ol>
            <li>In Vercel: set <code>VITE_API_URL</code> = <code>https://your-service.onrender.com/api/v1</code></li>
            <li>In Render: check the service is "Live" (not suspended)</li>
            <li>In Render environment variables: make sure <code>PORT</code> is not overriding the default</li>
          </ol>
        </div>
        <button className="btn-ghost" onClick={initApp} style={{ marginTop: 16 }}>
          Try again
        </button>
      </div>
    );
  }

  // ── Main app ──────────────────────────────────────────────────────────────
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="logo">
            <span className="logo-icon">◈</span>
            <span className="logo-text">DocSage</span>
          </div>
          <p className="logo-tagline">Smart Document QA</p>
        </div>

        <nav className="sidebar-nav">
          <button
            className={`nav-item ${view === "chat" ? "active" : ""}`}
            onClick={() => setView("chat")}
          >
            <span className="nav-icon">⬡</span> Chat
          </button>
          <button
            className={`nav-item ${view === "library" ? "active" : ""}`}
            onClick={() => setView("library")}
          >
            <span className="nav-icon">⬢</span> Library
            {documents.length > 0 && (
              <span className="nav-badge">{documents.length}</span>
            )}
          </button>
        </nav>

        <div className="sidebar-upload">
          <DocumentUploader onDocumentReady={handleDocumentReady} />
        </div>

        {documents.length > 0 && (
          <div className="active-docs">
            <p className="active-docs-label">Search scope</p>
            {documents
              .filter((d) => d.status === "ready")
              .map((doc) => (
                <button
                  key={doc.doc_id}
                  className={`doc-filter-chip ${
                    activeDocIds.includes(doc.doc_id) ? "selected" : ""
                  }`}
                  onClick={() => toggleDocActive(doc.doc_id)}
                  title={doc.filename}
                >
                  <span className="chip-dot" />
                  {doc.filename.length > 24
                    ? doc.filename.slice(0, 22) + "…"
                    : doc.filename}
                </button>
              ))}
            {activeDocIds.length > 0 && (
              <button
                className="clear-filter"
                onClick={() => setActiveDocIds([])}
              >
                Clear filter
              </button>
            )}
          </div>
        )}

        <div className="sidebar-footer">
          <SessionBadge sessionId={sessionId} />
        </div>
      </aside>

      <main className="main-content">
        {view === "chat" ? (
          <ChatInterface
            sessionId={sessionId}
            activeDocIds={activeDocIds.length > 0 ? activeDocIds : null}
            hasDocuments={documents.some((d) => d.status === "ready")}
          />
        ) : (
          <DocumentLibrary
            documents={documents}
            activeDocIds={activeDocIds}
            onToggleActive={toggleDocActive}
            onDelete={handleDocumentDelete}
          />
        )}
      </main>
    </div>
  );
}
