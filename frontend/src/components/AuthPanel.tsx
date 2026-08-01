import { useState, type FormEvent } from "react";
import { motion } from "framer-motion";
import { login, signup, setToken, type User } from "../api/client";
import "./AuthPanel.css";

type Props = {
  onAuth: (user: User) => void;
};

export function AuthPanel({ onAuth }: Props) {
  const [mode, setMode] = useState<"login" | "signup">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      const res = mode === "login" ? await login(email, password) : await signup(email, password);
      setToken(res.access_token);
      onAuth(res.user);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Authentication failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <motion.section
      className="auth"
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45 }}
    >
      <header className="auth__brand">
        <img src="/aegis.svg" alt="" width={32} height={32} />
        <span>Aegis</span>
      </header>

      <div className="auth__card">
        <h1>{mode === "login" ? "Welcome back" : "Create your account"}</h1>
        <p className="auth__lede">
          Each chat keeps its own uploads — answers only come from that conversation’s
          files.
        </p>

        <form className="auth__form" onSubmit={submit}>
          <label>
            Email
            <input
              type="email"
              autoComplete="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
            />
          </label>
          <label>
            Password
            <input
              type="password"
              autoComplete={mode === "login" ? "current-password" : "new-password"}
              required
              minLength={8}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="At least 8 characters"
            />
          </label>
          {error ? <div className="auth__error">{error}</div> : null}
          <button type="submit" className="btn-primary" disabled={busy}>
            {busy ? "Please wait…" : mode === "login" ? "Log in" : "Sign up"}
          </button>
        </form>

        <p className="auth__switch">
          {mode === "login" ? (
            <>
              New here?{" "}
              <button type="button" onClick={() => setMode("signup")}>
                Create an account
              </button>
            </>
          ) : (
            <>
              Already have an account?{" "}
              <button type="button" onClick={() => setMode("login")}>
                Log in
              </button>
            </>
          )}
        </p>
      </div>
    </motion.section>
  );
}
