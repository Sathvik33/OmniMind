import "./ChatSidebar.css";
import type { ChatSummary } from "../api/client";

type Props = {
  chats: ChatSummary[];
  activeChatId: number | null;
  userEmail: string;
  onSelect: (id: number) => void;
  onNew: () => void;
  onDelete: (id: number) => void;
  onLogout: () => void;
};

export function ChatSidebar({
  chats,
  activeChatId,
  userEmail,
  onSelect,
  onNew,
  onDelete,
  onLogout,
}: Props) {
  return (
    <aside className="sidebar">
      <div className="sidebar__top">
        <div className="sidebar__brand">
          <img src="/aegis.svg" alt="" width={24} height={24} />
          <strong>Aegis</strong>
        </div>
        <button type="button" className="sidebar__new" onClick={onNew}>
          New chat
        </button>
      </div>

      <nav className="sidebar__list" aria-label="Chat history">
        {chats.length === 0 ? (
          <p className="sidebar__empty">No chats yet</p>
        ) : (
          chats.map((c) => (
            <div
              key={c.id}
              className={`sidebar__item ${activeChatId === c.id ? "sidebar__item--active" : ""}`}
            >
              <button type="button" className="sidebar__item-btn" onClick={() => onSelect(c.id)}>
                {c.title || "New chat"}
              </button>
              <button
                type="button"
                className="sidebar__item-del"
                aria-label="Delete chat"
                onClick={(e) => {
                  e.stopPropagation();
                  onDelete(c.id);
                }}
              >
                ×
              </button>
            </div>
          ))
        )}
      </nav>

      <div className="sidebar__foot">
        <span className="sidebar__email" title={userEmail}>
          {userEmail}
        </span>
        <button type="button" className="ghost-btn" onClick={onLogout}>
          Log out
        </button>
      </div>
    </aside>
  );
}
