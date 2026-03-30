"use client";

import { useMemo, useState } from "react";

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: string[];
};

type Banner = {
  type: "error" | "warning" | "info";
  message: string;
};

const ROLE_COLLECTIONS: Record<string, string[]> = {
  employee: ["general"],
  finance: ["finance", "general"],
  engineering: ["engineering", "general"],
  marketing: ["marketing", "general"],
  hr: ["hr", "general"],
  c_level: ["general", "finance", "engineering", "marketing", "hr"],
};
const ALL_COLLECTIONS = ["general", "finance", "engineering", "marketing", "hr"];

export default function Home() {
  const apiBase = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [userId, setUserId] = useState("");
  const [role, setRole] = useState("");
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [banner, setBanner] = useState<Banner | null>(null);

  const roleCollections = useMemo(() => ROLE_COLLECTIONS[role] || [], [role]);

  const handleLogin = async () => {
    setBanner(null);
    setLoading(true);
    try {
      const response = await fetch(`${apiBase}/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });

      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || "Login failed");
      }

      const data = (await response.json()) as { user_id: string; role: string };
      setUserId(data.user_id);
      setRole(data.role);
      setBanner({
        type: "info",
        message: `Signed in as ${data.user_id} (${data.role}).`,
      });
    } catch (error) {
      setBanner({
        type: "error",
        message: error instanceof Error ? error.message : "Login failed",
      });
    } finally {
      setLoading(false);
    }
  };

  const handleSend = async () => {
    if (!query.trim()) return;
    if (!role) {
      setBanner({
        type: "warning",
        message: "Please log in to activate RBAC before querying.",
      });
      return;
    }

    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: query.trim(),
    };
    setMessages((prev) => [...prev, userMessage]);
    setQuery("");
    setLoading(true);
    setBanner(null);

    try {
      const response = await fetch(`${apiBase}/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-User-Id": userId || role,
        },
        body: JSON.stringify({ query: userMessage.content, user_role: role }),
      });

      const payload = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(payload.detail || "Query failed");
      }

      let answerText = payload.answer || "";
      if (answerText.includes("⚠️")) {
        const warningIndex = answerText.indexOf("⚠️");
        const warningMessage = answerText.slice(warningIndex).trim();
        answerText = answerText.slice(0, warningIndex).trim();
        setBanner({
          type: "warning",
          message: warningMessage.replace("⚠️", "").trim(),
        });
      }

      const assistantMessage: Message = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: answerText || "No response generated.",
        sources: payload.sources || [],
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      setBanner({
        type: "error",
        message: error instanceof Error ? error.message : "Query failed",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="relative flex flex-1 flex-col overflow-hidden">
      <div className="pointer-events-none absolute inset-0 bg-grid opacity-30" />

      <header className="relative z-10 flex items-center justify-between px-6 py-6 md:px-12">
        <div className="fade-up flex items-center gap-3">
          <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-teal-400/20 text-teal-200 shadow-[0_0_40px_rgba(45,212,191,0.35)]">
            <span className="font-mono text-sm">FB</span>
          </div>
          <div>
            <p className="text-xs uppercase tracking-[0.3em] text-slate-400">
              FinSolve
            </p>
            <h1 className="text-2xl font-semibold tracking-tight">FinBot</h1>
          </div>
        </div>
        <div className="fade-up-delayed flex items-center gap-2 text-xs uppercase tracking-[0.25em] text-slate-400">
          Advanced RAG
          <span className="rounded-full border border-teal-400/30 px-3 py-1 text-[10px] text-teal-200">
            live
          </span>
        </div>
      </header>

      <main className="relative z-10 flex flex-1 flex-col gap-6 px-6 pb-10 md:px-12 lg:flex-row">
        <section className="fade-up-more glass flex w-full flex-col gap-6 rounded-3xl p-6 lg:max-w-sm">
          <div>
            <p className="text-xs uppercase tracking-[0.25em] text-slate-400">
              Access Portal
            </p>
            <h2 className="text-xl font-semibold">Login</h2>
          </div>

          <div className="flex flex-col gap-3">
            <label className="text-xs uppercase tracking-[0.2em] text-slate-400">
              Username
            </label>
            <input
              value={username}
              onChange={(event) => setUsername(event.target.value)}
              className="rounded-2xl border border-slate-700/50 bg-slate-900/60 px-4 py-3 text-sm text-slate-100 outline-none transition focus:border-teal-400/60"
              placeholder="alice"
            />
            <label className="text-xs uppercase tracking-[0.2em] text-slate-400">
              Password
            </label>
            <input
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              type="password"
              className="rounded-2xl border border-slate-700/50 bg-slate-900/60 px-4 py-3 text-sm text-slate-100 outline-none transition focus:border-teal-400/60"
              placeholder="••••••••"
            />
            <button
              onClick={handleLogin}
              disabled={loading}
              className="rounded-2xl bg-teal-400/90 px-4 py-3 text-sm font-semibold text-slate-950 transition hover:bg-teal-300 disabled:opacity-60"
            >
              {loading ? "Authenticating..." : "Login"}
            </button>
            <p className="text-xs text-slate-500">
              Demo users: <span className="font-mono">alice / bob / carol / dave / erin</span>
            </p>
          </div>

          <div className="rounded-2xl border border-slate-700/50 bg-slate-900/40 p-4">
            <p className="text-xs uppercase tracking-[0.2em] text-slate-400">
              Active Role
            </p>
            <p className="mt-2 text-lg font-semibold">
              {role ? role.replace("_", " ") : "Not signed in"}
            </p>
            <p className="mt-1 text-xs text-slate-400">
              {userId ? `User ID: ${userId}` : "Authenticate to unlock data."}
            </p>
          </div>

          <div>
            <p className="text-xs uppercase tracking-[0.2em] text-slate-400">
              RBAC Access
            </p>
            <div className="mt-3 flex flex-wrap gap-2">
              {ALL_COLLECTIONS.map((collection) => {
                const enabled = roleCollections.includes(collection);
                return (
                  <span
                    key={collection}
                    className={`rounded-full border px-3 py-1 text-xs uppercase tracking-[0.2em] ${
                      enabled
                        ? "border-amber-300/60 bg-amber-300/10 text-amber-200"
                        : "border-slate-700/60 text-slate-500"
                    }`}
                  >
                    {collection}
                  </span>
                );
              })}
            </div>
          </div>
        </section>

        <section className="glass flex min-h-[640px] w-full flex-1 flex-col rounded-3xl p-6">
          <div className="flex flex-col gap-2">
            <p className="text-xs uppercase tracking-[0.25em] text-slate-400">
              Secure Chat
            </p>
            <div className="flex items-center justify-between">
              <h2 className="text-2xl font-semibold">Ask FinBot</h2>
              <span className="text-xs text-slate-400">
                Routed by intent + RBAC
              </span>
            </div>
          </div>

          {banner && (
            <div
              className={`mt-4 rounded-2xl border px-4 py-3 text-sm ${
                banner.type === "error"
                  ? "border-rose-400/60 bg-rose-500/10 text-rose-200"
                  : banner.type === "warning"
                  ? "border-amber-300/60 bg-amber-400/10 text-amber-100"
                  : "border-sky-300/60 bg-sky-500/10 text-sky-100"
              }`}
            >
              {banner.message}
            </div>
          )}

          <div className="mt-4 flex flex-1 flex-col gap-4 overflow-y-auto rounded-3xl border border-slate-800/60 bg-slate-950/40 p-4">
            {messages.length === 0 ? (
              <div className="flex h-full flex-col items-center justify-center text-center text-sm text-slate-500">
                <p className="text-base font-semibold text-slate-200">
                  Start with a secure question.
                </p>
                <p className="mt-2 max-w-sm">
                  Example: “Show me the Q3 earnings summary for FinSolve.”
                </p>
              </div>
            ) : (
              messages.map((message) => (
                <div
                  key={message.id}
                  className={`rounded-2xl border px-4 py-3 text-sm ${
                    message.role === "user"
                      ? "self-end border-teal-400/50 bg-teal-500/10 text-teal-50"
                      : "self-start border-slate-700/60 bg-slate-900/60 text-slate-100"
                  }`}
                >
                  <p className="whitespace-pre-wrap">{message.content}</p>
                  {message.sources && message.sources.length > 0 && (
                    <div className="mt-3 text-xs uppercase tracking-[0.2em] text-slate-400">
                      Sources:{" "}
                      <span className="normal-case text-slate-300">
                        {message.sources.join(", ")}
                      </span>
                    </div>
                  )}
                </div>
              ))
            )}
          </div>

          <div className="mt-4 flex flex-col gap-3 rounded-3xl border border-slate-800/70 bg-slate-950/60 p-4">
            <textarea
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              rows={3}
              placeholder="Ask about finance, engineering, marketing, or general ops..."
              className="w-full resize-none bg-transparent text-sm text-slate-100 outline-none placeholder:text-slate-500"
            />
            <div className="flex items-center justify-between">
              <p className="text-xs text-slate-500">
                Guardrails enforce PII, injection, and role-based access.
              </p>
              <button
                onClick={handleSend}
                disabled={loading}
                className="rounded-full bg-amber-300 px-5 py-2 text-xs font-semibold uppercase tracking-[0.2em] text-slate-950 transition hover:bg-amber-200 disabled:opacity-60"
              >
                {loading ? "Routing..." : "Send"}
              </button>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
