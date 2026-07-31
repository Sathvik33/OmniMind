import { motion } from "framer-motion";
import "./Landing.css";

type Props = {
  onEnter: () => void;
};

export function Landing({ onEnter }: Props) {
  return (
    <section className="landing">
      <motion.header
        className="landing__brand"
        initial={{ opacity: 0, y: -8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.55, ease: [0.22, 1, 0.36, 1] }}
      >
        <img src="/aegis.svg" alt="" width={32} height={32} />
        <span>Aegis</span>
      </motion.header>

      <div className="landing__stage">
        <div className="landing__copy">
          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.08, ease: [0.22, 1, 0.36, 1] }}
          >
            Aegis
          </motion.h1>
          <motion.p
            className="landing__lede"
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.65, delay: 0.2, ease: [0.22, 1, 0.36, 1] }}
          >
            Ask questions against your own files — answers stay tied to what
            you uploaded.
          </motion.p>
          <motion.div
            className="landing__cta"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.32, ease: [0.22, 1, 0.36, 1] }}
          >
            <button type="button" className="btn-primary" onClick={onEnter}>
              Start a conversation
            </button>
            <span className="landing__hint">PDF, slides, images, video</span>
          </motion.div>
        </div>

        <motion.div
          className="landing__desk"
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.75, delay: 0.28, ease: [0.22, 1, 0.36, 1] }}
          aria-hidden
        >
          <div className="desk-sheet desk-sheet--back" />
          <div className="desk-sheet desk-sheet--mid" />
          <div className="desk-sheet desk-sheet--front">
            <span className="desk-line" />
            <span className="desk-line" />
            <span className="desk-line" />
            <span className="desk-line" />
          </div>
        </motion.div>
      </div>
    </section>
  );
}
