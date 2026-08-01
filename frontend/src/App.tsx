import { useCallback, useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { AmbientBg } from "./components/AmbientBg";
import { AuthPanel } from "./components/AuthPanel";
import { ChatSidebar } from "./components/ChatSidebar";
import { Landing } from "./components/Landing";
import { Workspace } from "./components/Workspace";
import {
  createChat,
  deleteChat,
  fetchMe,
  getToken,
  listChats,
  setToken,
  type ChatSummary,
  type User,
} from "./api/client";
import "./AppShell.css";

type View = "landing" | "auth" | "app";

export default function App() {
  const [view, setView] = useState<View>("landing");
  const [user, setUser] = useState<User | null>(null);
  const [chats, setChats] = useState<ChatSummary[]>([]);
  const [activeChatId, setActiveChatId] = useState<number | null>(null);
  const [bootError, setBootError] = useState<string | null>(null);

  const refreshChats = useCallback(async (preferId?: number | null) => {
    const rows = await listChats();
    setChats(rows);
    setActiveChatId((prev) => {
      if (preferId != null && rows.some((c) => c.id === preferId)) return preferId;
      if (prev != null && rows.some((c) => c.id === prev)) return prev;
      return rows[0]?.id ?? null;
    });
    return rows;
  }, []);

  useEffect(() => {
    const token = getToken();
    if (!token) return;
    fetchMe()
      .then(async (u) => {
        setUser(u);
        setView("app");
        const rows = await refreshChats();
        if (!rows.length) {
          const created = await createChat();
          await refreshChats(created.id);
        }
      })
      .catch(() => {
        setToken(null);
      });
  }, [refreshChats]);

  const enterApp = async (u: User) => {
    setUser(u);
    setView("app");
    setBootError(null);
    try {
      const rows = await refreshChats();
      if (!rows.length) {
        const created = await createChat();
        await refreshChats(created.id);
      }
    } catch (e) {
      setBootError(e instanceof Error ? e.message : "Could not load chats");
    }
  };

  const onNewChat = async () => {
    try {
      const created = await createChat();
      await refreshChats(created.id);
    } catch (e) {
      setBootError(e instanceof Error ? e.message : "Could not create chat");
    }
  };

  const onDeleteChat = async (id: number) => {
    const title = chats.find((c) => c.id === id)?.title || "this chat";
    const ok = window.confirm(
      `Delete “${title}”?\n\nThis permanently removes the chat, uploads, embeddings, and MinIO files.`,
    );
    if (!ok) return;
    try {
      setBootError(null);
      await deleteChat(id);
      const rows = await refreshChats(activeChatId === id ? null : activeChatId);
      if (!rows.length) {
        const created = await createChat();
        await refreshChats(created.id);
      }
    } catch (e) {
      setBootError(e instanceof Error ? e.message : "Could not delete chat");
    }
  };

  const logout = () => {
    setToken(null);
    setUser(null);
    setChats([]);
    setActiveChatId(null);
    setView("landing");
  };

  return (
    <div className="app-shell">
      <AmbientBg />
      <div className="content-layer">
        <AnimatePresence mode="wait">
          {view === "landing" ? (
            <motion.div
              key="landing"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.3 }}
            >
              <Landing onEnter={() => setView("auth")} />
            </motion.div>
          ) : null}

          {view === "auth" ? (
            <motion.div
              key="auth"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.35 }}
            >
              <AuthPanel onAuth={enterApp} />
            </motion.div>
          ) : null}

          {view === "app" && user ? (
            <motion.div
              key="app"
              className="app-layout"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.35 }}
            >
              <ChatSidebar
                chats={chats}
                activeChatId={activeChatId}
                userEmail={user.email}
                onSelect={setActiveChatId}
                onNew={onNewChat}
                onDelete={onDeleteChat}
                onLogout={logout}
              />
              <div className="app-main">
                {bootError ? <div className="workspace__error">{bootError}</div> : null}
                {activeChatId != null ? (
                  <Workspace
                    chatId={activeChatId}
                    onChatUpdated={() => refreshChats(activeChatId)}
                  />
                ) : (
                  <p className="empty-chat__text" style={{ padding: "3rem" }}>
                    Create a chat to get started.
                  </p>
                )}
              </div>
            </motion.div>
          ) : null}
        </AnimatePresence>
      </div>
    </div>
  );
}
