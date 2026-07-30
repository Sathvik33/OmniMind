import { AnimatePresence, motion } from "framer-motion";
import "./MessageList.css";

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
};

type Props = {
  messages: ChatMessage[];
  streaming?: boolean;
};

export function MessageList({ messages, streaming }: Props) {
  if (messages.length === 0) {
    return (
      <motion.div
        className="empty-chat"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <p className="empty-chat__title">Ready when you are</p>
        <p className="empty-chat__text">
          Attach a PDF, image, or video, then ask a question grounded in that content.
        </p>
      </motion.div>
    );
  }

  return (
    <div className="message-list">
      <AnimatePresence initial={false}>
        {messages.map((msg, i) => (
          <motion.article
            key={msg.id}
            className={`bubble bubble--${msg.role}`}
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.35, delay: Math.min(i * 0.03, 0.2), ease: [0.22, 1, 0.36, 1] }}
          >
            <span className="bubble__role">{msg.role === "user" ? "You" : "AEGIS"}</span>
            <div className="bubble__body">
              {msg.content}
              {streaming && i === messages.length - 1 && msg.role === "assistant" ? (
                <span className="caret" />
              ) : null}
            </div>
          </motion.article>
        ))}
      </AnimatePresence>
    </div>
  );
}
