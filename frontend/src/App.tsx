import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { AmbientBg } from "./components/AmbientBg";
import { Landing } from "./components/Landing";
import { Workspace } from "./components/Workspace";

export default function App() {
  const [view, setView] = useState<"landing" | "workspace">("landing");

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
              <Landing onEnter={() => setView("workspace")} />
            </motion.div>
          ) : (
            <motion.div
              key="workspace"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
            >
              <Workspace onBack={() => setView("landing")} />
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
