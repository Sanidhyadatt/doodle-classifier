"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { io, type Socket } from "socket.io-client";

import DrawingCanvas, { type DrawingCanvasRef } from "@/components/DrawingCanvas";
import TeachModal from "@/components/TeachModal";

const API_BASE = "http://localhost:8000";
const ARENA_ROOM_ID = "arena-lobby";

type UserProfile = {
  id: number;
  username: string;
  display_name: string;
  xp: number;
  coins: number;
  level: number;
  streak: number;
  predictions_count: number;
  training_count: number;
  multiplayer_count: number;
  wiki_lookups: number;
  wins: number;
  losses: number;
  next_level_xp: number;
  level_progress: number;
  achievements: AchievementItem[];
};

type LeaderboardItem = {
  username: string;
  display_name: string;
  level: number;
  xp: number;
  coins: number;
  streak: number;
};

type MissionItem = {
  id: string;
  title: string;
  description: string;
  progress: number;
  target: number;
  reward: string;
};

type AchievementItem = {
  id: string;
  title: string;
  description: string;
  earned: boolean;
};

type ActivityItem = {
  kind: string;
  detail: string;
  created_at: string;
};

type ModelStatus = {
  ready: boolean;
  classes: string[];
};

type DashboardData = {
  user: UserProfile;
  leaderboard: LeaderboardItem[];
  missions: MissionItem[];
  achievements: AchievementItem[];
  recent_activity: ActivityItem[];
  correct_guesses: CorrectGuessItem[];
  model_status: ModelStatus;
};

type CorrectGuessItem = {
  guess_id: number;
  prediction: string;
  confidence: number;
  created_at: string;
};

type PredictResponse = {
  guess_id?: number;
  prediction: string;
  confidence: number;
  definition: string;
  user?: UserProfile;
  achievements?: AchievementItem[];
};

type GuessFeedbackResponse = {
  message: string;
  user: UserProfile;
  correct_guesses: CorrectGuessItem[];
};

type QuickDrawTrainResponse = {
  message: string;
  imported_total: number;
  imported_per_class: Record<string, number>;
  total_samples: number;
  unique_classes: number;
  model_ready: boolean;
  user: UserProfile;
  dashboard: DashboardData;
};

type AuthMode = "login" | "register";

type AuthResponse = {
  message: string;
  user: UserProfile;
  dashboard: DashboardData;
};

function formatActivity(kind: string) {
  switch (kind) {
    case "register":
      return "Joined the arena";
    case "login":
      return "Signed in";
    case "prediction":
      return "Made a prediction";
    case "training":
      return "Taught a new class";
    case "multiplayer":
      return "Entered a battle room";
    case "battle_guess":
      return "Broadcast an AI guess";
    default:
      return kind.replaceAll("_", " ");
  }
}

function useSocketMessages(roomId: string | null, enabled: boolean) {
  const [messages, setMessages] = useState<string[]>([]);
  const [remoteStroke, setRemoteStroke] = useState<{ x: number; y: number } | null>(null);
  const socketRef = useRef<Socket | null>(null);

  useEffect(() => {
    if (!enabled || !roomId) {
      socketRef.current?.disconnect();
      socketRef.current = null;
      return;
    }

    const socket = io(API_BASE, {
      transports: ["websocket"],
      withCredentials: true,
    });

    socketRef.current = socket;

    socket.on("connect", () => {
      socket.emit("join_room", { room_id: roomId });
      setMessages((prev) => [...prev, `Connected to ${roomId}`]);
    });

    socket.on("ai_chat_message", (payload: {
      message?: string;
      prediction?: string;
      confidence?: number;
      definition?: string;
      user?: { display_name?: string } | null;
    }) => {
      const message = payload.message
        ?? (payload.prediction
          ? `AI guess: ${payload.prediction}${
              typeof payload.confidence === "number"
                ? ` (${(payload.confidence * 100).toFixed(1)}%)`
                : ""
            }`
          : "Arena message received");

      const decoratedMessage = payload.definition
        ? `${message} · ${payload.definition}`
        : message;

      const userDisplayName = payload.user?.display_name;
      if (typeof userDisplayName === "string" && userDisplayName.length > 0) {
        setMessages((prev) => [...prev, `${userDisplayName}: ${decoratedMessage}`]);
        return;
      }

      setMessages((prev) => [...prev, decoratedMessage]);
    });

    socket.on("draw_stroke", (payload: { x?: number; y?: number }) => {
      if (typeof payload?.x === "number" && typeof payload?.y === "number") {
        setRemoteStroke({ x: payload.x, y: payload.y });
      }
    });

    socket.on("disconnect", () => {
      setMessages((prev) => [...prev, "Disconnected from server"]);
    });

    return () => {
      socket.disconnect();
      socketRef.current = null;
    };
  }, [enabled, roomId]);

  return { socketRef, messages, setMessages, remoteStroke };
}

export default function Home() {
  const canvasRef = useRef<DrawingCanvasRef>(null);
  const [session, setSession] = useState<DashboardData | null>(null);
  const [authMode, setAuthMode] = useState<AuthMode>("login");
  const [username, setUsername] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [password, setPassword] = useState("");
  const [authLoading, setAuthLoading] = useState(true);
  const [authSubmitting, setAuthSubmitting] = useState(false);
  const [authError, setAuthError] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const [isTeachModalOpen, setIsTeachModalOpen] = useState(false);
  const [quickdrawClasses, setQuickdrawClasses] = useState<string[]>([]);
  const [selectedQuickdrawClasses, setSelectedQuickdrawClasses] = useState<string[]>([]);
  const [quickdrawSamples, setQuickdrawSamples] = useState(40);
  const [quickdrawLoading, setQuickdrawLoading] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);
  const [prediction, setPrediction] = useState("No guess yet");
  const [lastGuessId, setLastGuessId] = useState<number | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [definition, setDefinition] = useState(
    "Log in to access the arena, then draw an object to begin training your personal model.",
  );
  const [isSamplesModalOpen, setIsSamplesModalOpen] = useState(false);
  const [viewingClass, setViewingClass] = useState<string | null>(null);
  const [classSamples, setClassSamples] = useState<number[][][]>([]);
  const [samplesLoading, setSamplesLoading] = useState(false);

  const { socketRef, messages, setMessages, remoteStroke } = useSocketMessages(
    ARENA_ROOM_ID,
    Boolean(session),
  );

  const refreshDashboard = useCallback(async (options?: { silent?: boolean }) => {
    if (!options?.silent) {
      setAuthLoading(true);
    }

    try {
      const response = await fetch(`${API_BASE}/dashboard`, {
        credentials: "include",
      });

      if (!response.ok) {
        throw new Error("Not authenticated");
      }

      const dashboard = (await response.json()) as DashboardData;
      setSession(dashboard);
      setMessages((prev) =>
        prev.length > 0 ? prev : [`Welcome back, ${dashboard.user.display_name}`],
      );
      return dashboard;
    } catch {
      setSession(null);
      return null;
    } finally {
      if (!options?.silent) {
        setAuthLoading(false);
      }
    }
  }, [setMessages]);

  useEffect(() => {
    void refreshDashboard();
  }, [refreshDashboard]);

  useEffect(() => {
    if (!remoteStroke) return;
    canvasRef.current?.drawRemotePoint(remoteStroke.x, remoteStroke.y);
  }, [remoteStroke]);

  useEffect(() => {
    if (!session) return;

    const loadQuickDrawClasses = async () => {
      try {
        const response = await fetch(`${API_BASE}/quickdraw/classes`, {
          credentials: "include",
        });
        if (!response.ok) return;
        const body = (await response.json()) as { classes?: string[] };
        if (Array.isArray(body.classes)) {
          setQuickdrawClasses(body.classes);
          if (body.classes.length > 0 && selectedQuickdrawClasses.length === 0) {
            setSelectedQuickdrawClasses(body.classes.slice(0, 3));
          }
        }
      } catch {
        // Keep UI usable even if class list fetch fails.
      }
    };

    void loadQuickDrawClasses();
  }, [session, selectedQuickdrawClasses.length]);

  useEffect(() => {
    if (!toast) return;
    const timeout = window.setTimeout(() => setToast(null), 3000);
    return () => window.clearTimeout(timeout);
  }, [toast]);

  const levelProgress = useMemo(() => {
    if (!session) return 0;
    return Math.max(0, Math.min(100, session.user.level_progress));
  }, [session]);

  const handleAuthSubmit = useCallback(async () => {
    setAuthSubmitting(true);
    setAuthError(null);

    try {
      const endpoint = authMode === "login" ? "login" : "register";
      const payload: { username: string; password: string; display_name?: string } = {
        username,
        password,
      };

      if (authMode === "register" && displayName.trim()) {
        payload.display_name = displayName.trim();
      }

      const response = await fetch(`${API_BASE}/auth/${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify(payload),
      });

      const body = (await response.json()) as AuthResponse | { detail?: string };
      if (!response.ok) {
        throw new Error("detail" in body ? String(body.detail ?? "Authentication failed") : "Authentication failed");
      }

      if ("dashboard" in body) {
        const authBody = body as AuthResponse;
        setSession(authBody.dashboard);
        setPrediction("No guess yet");
        setLastGuessId(null);
        setConfidence(null);
        setDefinition("Your dashboard is ready. Draw to train, guess, and battle.");
        setToast(authBody.message);
        setMessages([`Signed in as ${authBody.user.display_name}`]);
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "Authentication failed";
      setAuthError(message);
    } finally {
      setAuthSubmitting(false);
      setAuthLoading(false);
    }
  }, [authMode, displayName, password, setMessages, username]);

  const handleLogout = useCallback(async () => {
    await fetch(`${API_BASE}/auth/logout`, {
      method: "POST",
      credentials: "include",
    });
    socketRef.current?.disconnect();
    setSession(null);
    setMessages([]);
    setPrediction("No guess yet");
    setLastGuessId(null);
    setConfidence(null);
    setDefinition("Log in to continue your arena run.");
    setToast("Logged out successfully");
  }, [setMessages, socketRef]);

  const handleGuess = useCallback(async () => {
    if (!session) return;

    const pixels = canvasRef.current?.getGrayscalePixels() ?? [];
    if (!pixels.length) return;

    setIsPredicting(true);
    socketRef.current?.emit("trigger_ai_guess", {
      room_id: ARENA_ROOM_ID,
      pixel_data: pixels,
    });

    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ pixel_data: pixels }),
      });

      const body = (await response.json()) as PredictResponse | { detail?: string };
      if (!response.ok) {
        throw new Error("detail" in body ? String(body.detail ?? "Prediction failed") : "Prediction failed");
      }

      if ("prediction" in body) {
        setPrediction(body.prediction);
        setLastGuessId(typeof body.guess_id === "number" ? body.guess_id : null);
        setConfidence(body.confidence);
        setDefinition(body.definition);
        const updatedUser = body.user;
        if (updatedUser) {
          setSession((current) => {
            if (!current) return current;
            return {
              ...current,
              user: updatedUser,
              achievements: body.achievements ?? current.achievements,
            };
          });
        }
        void refreshDashboard({ silent: true });
      }
    } catch {
      setPrediction("Prediction unavailable");
      setConfidence(null);
      setDefinition("Could not reach backend predict endpoint.");
    } finally {
      setIsPredicting(false);
    }
  }, [refreshDashboard, session, socketRef]);

  const handleMultiplayerBattle = useCallback(() => {
    if (!session) return;
    socketRef.current?.emit("join_room", { room_id: ARENA_ROOM_ID });
    setToast("Battle room joined");
  }, [session, socketRef]);

  const handleStrokePoint = useCallback(
    (x: number, y: number) => {
      if (!session) return;
      socketRef.current?.emit("draw_stroke", {
        x,
        y,
        room_id: ARENA_ROOM_ID,
      });
    },
    [session, socketRef],
  );

  const handleModalTrained = useCallback(
    (className: string) => {
      setToast(`Taught ${className}`);
      void refreshDashboard({ silent: true });
    },
    [refreshDashboard],
  );

  const handleToggleQuickdrawClass = useCallback((className: string) => {
    setSelectedQuickdrawClasses((current) =>
      current.includes(className)
        ? current.filter((item) => item !== className)
        : [...current, className],
    );
  }, []);

  const handleQuickdrawTrain = useCallback(async () => {
    if (!session || selectedQuickdrawClasses.length === 0) return;

    setQuickdrawLoading(true);
    try {
      const response = await fetch(`${API_BASE}/quickdraw/train`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({
          class_names: selectedQuickdrawClasses,
          samples_per_class: quickdrawSamples,
        }),
      });

      const body = (await response.json()) as QuickDrawTrainResponse | { detail?: string };
      if (!response.ok) {
        throw new Error("detail" in body ? String(body.detail ?? "QuickDraw training failed") : "QuickDraw training failed");
      }

      const quickdrawBody = body as QuickDrawTrainResponse;
      setSession(quickdrawBody.dashboard);
      setToast(
        `QuickDraw imported ${quickdrawBody.imported_total} samples from ${Object.keys(quickdrawBody.imported_per_class).length} classes`,
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : "QuickDraw training failed";
      setToast(message);
    } finally {
      setQuickdrawLoading(false);
    }
  }, [quickdrawSamples, selectedQuickdrawClasses, session]);

  const handleGuessFeedback = useCallback(
    async (isCorrect: boolean) => {
      if (!session || !lastGuessId) return;

      try {
        const response = await fetch(`${API_BASE}/predict/feedback`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          credentials: "include",
          body: JSON.stringify({ guess_id: lastGuessId, is_correct: isCorrect }),
        });

        const body = (await response.json()) as GuessFeedbackResponse | { detail?: string };
        if (!response.ok) {
          throw new Error("detail" in body ? String(body.detail ?? "Feedback failed") : "Feedback failed");
        }

        const feedbackBody = body as GuessFeedbackResponse;
        setSession((current) =>
          current
            ? {
                ...current,
                user: feedbackBody.user,
                correct_guesses: feedbackBody.correct_guesses,
              }
            : current,
        );
        setToast(feedbackBody.message);
      } catch {
        setToast("Failed to record feedback");
      }
    },
    [lastGuessId, session],
  );


  const handleViewSamples = useCallback(async (className: string) => {
    setViewingClass(className);
    setSamplesLoading(true);
    setIsSamplesModalOpen(true);
    setClassSamples([]);

    try {
      const response = await fetch(`${API_BASE}/training_samples/${className}`, {
        credentials: "include",
      });
      if (!response.ok) throw new Error("Failed to fetch samples");
      const body = (await response.json()) as { samples: number[][][] };
      if (Array.isArray(body.samples)) {
        setClassSamples(body.samples);
      }
    } catch {
      setToast("Could not load training samples");
    } finally {
      setSamplesLoading(false);
    }
  }, []);

  const handleRemoveClass = useCallback(async (className: string) => {
    if (!confirm(`Are you sure you want to delete the class "${className}" and all its training samples? This will retrain your personal model.`)) {
      return;
    }

    try {
      const response = await fetch(`${API_BASE}/training_data/${className}`, {
        method: "DELETE",
        credentials: "include",
      });

      if (!response.ok) {
        const body = (await response.json()) as { detail?: string };
        throw new Error(body.detail ?? "Failed to remove class");
      }

      const body = (await response.json()) as { message: string; dashboard: DashboardData };
      setSession(body.dashboard);
      setToast(body.message);
    } catch (error) {
      setToast(error instanceof Error ? error.message : "Cleanup failed");
    }
  }, []);

  const authScreen = (
    <div className="min-h-screen px-4 py-8 text-slate-900">
      <div className="mx-auto grid min-h-[calc(100vh-4rem)] w-full max-w-6xl gap-6 lg:grid-cols-[1.1fr_0.9fr]">
        <section className="rounded-[2rem] border border-white/70 bg-white/90 p-8 shadow-[0_20px_80px_rgba(15,23,42,0.08)] backdrop-blur">
          <div className="inline-flex items-center gap-2 rounded-full border border-indigo-200 bg-indigo-50 px-3 py-1 text-xs font-semibold uppercase tracking-[0.2em] text-indigo-700">
            NeuroSketch Arena
          </div>
          <h1 className="mt-5 max-w-xl text-5xl font-semibold tracking-tight text-slate-950 sm:text-6xl">
            A gamified doodle lab where every sketch levels you up.
          </h1>
          <p className="mt-5 max-w-2xl text-lg leading-8 text-slate-600">
            Log in to unlock the protected arena, train new objects, battle in multiplayer rooms,
            and climb the leaderboard through real drawing, guessing, and teaching.
          </p>

          <div className="mt-8 grid gap-4 sm:grid-cols-2">
            {[
              ["Live model", "Pretrained SVM, classification, and Wikipedia coach"],
              ["Progression", "XP, coins, streaks, missions, achievements"],
              ["Arena mode", "Socket.IO sketch battles and room chat"],
              ["Training loop", "Teach custom objects from your own drawings"],
            ].map(([title, text]) => (
              <div key={title} className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                <p className="text-sm font-semibold text-slate-900">{title}</p>
                <p className="mt-1 text-sm text-slate-600">{text}</p>
              </div>
            ))}
          </div>

          <div className="mt-8 grid gap-4 sm:grid-cols-3">
            {[
              ["Sign in", "Use an existing arena account"],
              ["Draw", "Guess or teach a sketch"],
              ["Battle", "Join the shared multiplayer room"],
            ].map(([title, text]) => (
              <div key={title} className="rounded-2xl bg-gradient-to-br from-slate-900 to-indigo-900 p-4 text-white shadow-lg">
                <p className="text-sm font-semibold uppercase tracking-[0.15em] text-indigo-200">{title}</p>
                <p className="mt-3 text-sm text-slate-100/90">{text}</p>
              </div>
            ))}
          </div>
        </section>

        <section className="flex items-center">
          <div className="w-full rounded-[2rem] border border-slate-200 bg-white p-6 shadow-[0_20px_80px_rgba(15,23,42,0.08)]">
            <div className="flex items-center justify-between gap-3">
              <div>
                <p className="text-sm font-semibold uppercase tracking-[0.2em] text-indigo-700">
                  Secure Access
                </p>
                <h2 className="mt-2 text-2xl font-semibold text-slate-950">
                  {authMode === "login" ? "Welcome back" : "Create your arena profile"}
                </h2>
              </div>
              <div className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-xs font-medium text-slate-600">
                HTTPS-ready cookie session
              </div>
            </div>

            <div className="mt-6 flex rounded-full bg-slate-100 p-1">
              <button
                type="button"
                onClick={() => setAuthMode("login")}
                className={`flex-1 rounded-full px-4 py-2 text-sm font-semibold transition ${
                  authMode === "login"
                    ? "bg-white text-slate-950 shadow-sm"
                    : "text-slate-500 hover:text-slate-800"
                }`}
              >
                Login
              </button>
              <button
                type="button"
                onClick={() => setAuthMode("register")}
                className={`flex-1 rounded-full px-4 py-2 text-sm font-semibold transition ${
                  authMode === "register"
                    ? "bg-white text-slate-950 shadow-sm"
                    : "text-slate-500 hover:text-slate-800"
                }`}
              >
                Register
              </button>
            </div>

            <div className="mt-6 space-y-4">
              <label className="block text-sm font-medium text-slate-700">
                Username
                <input
                  value={username}
                  onChange={(event) => setUsername(event.target.value)}
                  autoComplete="username"
                  className="mt-2 w-full rounded-2xl border border-slate-300 bg-white px-4 py-3 text-sm outline-none ring-indigo-200 focus:ring-4"
                  placeholder="yourname"
                />
              </label>

              {authMode === "register" ? (
                <label className="block text-sm font-medium text-slate-700">
                  Display Name
                  <input
                    value={displayName}
                    onChange={(event) => setDisplayName(event.target.value)}
                    autoComplete="nickname"
                    className="mt-2 w-full rounded-2xl border border-slate-300 bg-white px-4 py-3 text-sm outline-none ring-indigo-200 focus:ring-4"
                    placeholder="Sketch Captain"
                  />
                </label>
              ) : null}

              <label className="block text-sm font-medium text-slate-700">
                Password
                <input
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  type="password"
                  autoComplete={authMode === "login" ? "current-password" : "new-password"}
                  className="mt-2 w-full rounded-2xl border border-slate-300 bg-white px-4 py-3 text-sm outline-none ring-indigo-200 focus:ring-4"
                  placeholder="••••••••"
                />
              </label>
            </div>

            {authError ? (
              <div className="mt-5 rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                {authError}
              </div>
            ) : null}

            <div className="mt-6 flex items-center gap-3">
              <button
                type="button"
                onClick={handleAuthSubmit}
                disabled={authSubmitting}
                className="rounded-2xl bg-indigo-600 px-5 py-3 text-sm font-semibold text-white transition hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-70"
              >
                {authSubmitting ? "Working..." : authMode === "login" ? "Enter Arena" : "Create Account"}
              </button>
              <div className="text-xs text-slate-500">
                Session is required to view the dashboard, use the model, and join multiplayer.
              </div>
            </div>
          </div>
        </section>
      </div>
    </div>
  );

  const loadingScreen = (
    <div className="flex min-h-screen items-center justify-center px-4 text-slate-700">
      <div className="rounded-3xl border border-white/70 bg-white/90 px-6 py-5 shadow-[0_20px_80px_rgba(15,23,42,0.08)] backdrop-blur">
        <p className="text-sm font-semibold uppercase tracking-[0.2em] text-indigo-700">
          NeuroSketch Arena
        </p>
        <p className="mt-2 text-lg font-semibold text-slate-950">
          Loading your session and arena state...
        </p>
      </div>
    </div>
  );

  const dashboardScreen = session ? (
    <div className="min-h-screen px-4 py-6 text-slate-900">
      <div className="mx-auto flex w-full max-w-7xl flex-col gap-6">
        <header className="rounded-[2rem] border border-white/70 bg-white/90 px-6 py-5 shadow-[0_20px_80px_rgba(15,23,42,0.08)] backdrop-blur">
          <div className="flex flex-col gap-5 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <div className="inline-flex items-center gap-2 rounded-full border border-indigo-200 bg-indigo-50 px-3 py-1 text-xs font-semibold uppercase tracking-[0.2em] text-indigo-700">
                NeuroSketch Arena
              </div>
              <h1 className="mt-3 text-3xl font-semibold tracking-tight text-slate-950">
                {session.user.display_name}&apos;s command center
              </h1>
              <p className="mt-1 text-sm text-slate-500">
                Level {session.user.level} · {session.user.xp} XP · {session.user.coins} coins · streak {session.user.streak} days
              </p>
            </div>

            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              {[
                ["Level", String(session.user.level)],
                ["XP", String(session.user.xp)],
                ["Coins", String(session.user.coins)],
                ["Battles", String(session.user.multiplayer_count)],
              ].map(([label, value]) => (
                <div key={label} className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-center">
                  <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-500">{label}</p>
                  <p className="text-lg font-semibold text-slate-950">{value}</p>
                </div>
              ))}
            </div>
          </div>

          <div className="mt-5">
            <div className="mb-2 flex items-center justify-between text-xs font-medium text-slate-500">
              <span>Level Progress</span>
              <span>{session.user.xp} / {session.user.next_level_xp} XP</span>
            </div>
            <div className="h-3 rounded-full bg-slate-200">
              <div
                className="h-3 rounded-full bg-gradient-to-r from-indigo-600 to-indigo-400 transition-all"
                style={{ width: `${levelProgress}%` }}
              />
            </div>
          </div>

          <div className="mt-6 flex flex-wrap items-center gap-3">
            <button
              type="button"
              onClick={() => setIsTeachModalOpen(true)}
              className="rounded-2xl border border-indigo-200 bg-indigo-50 px-4 py-2 text-sm font-semibold text-indigo-700 transition hover:bg-indigo-100"
            >
              Teach New Object
            </button>
            <button
              type="button"
              onClick={handleMultiplayerBattle}
              className="rounded-2xl bg-slate-900 px-4 py-2 text-sm font-semibold text-white transition hover:bg-slate-700"
            >
              Multiplayer Battle
            </button>
            <button
              type="button"
              onClick={handleLogout}
              className="rounded-2xl border border-slate-300 bg-white px-4 py-2 text-sm font-semibold text-slate-700 transition hover:bg-slate-50"
            >
              Logout
            </button>
          </div>
        </header>

        <main className="grid grid-cols-1 gap-6 xl:grid-cols-[1.1fr_0.9fr]">
          <section className="space-y-6">
            <article className="rounded-[2rem] border border-slate-200 bg-white p-6 shadow-[0_20px_80px_rgba(15,23,42,0.08)]">
              <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <p className="text-sm font-semibold uppercase tracking-[0.2em] text-indigo-700">
                    Battle Canvas
                  </p>
                  <h2 className="mt-2 text-2xl font-semibold text-slate-950">
                    Draw, guess, train, repeat.
                  </h2>
                  <p className="mt-1 text-sm text-slate-500">
                    Every stroke can earn XP, advance missions, and influence your arena rank.
                  </p>
                </div>
                <div className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-600">
                  <p className="font-semibold text-slate-900">
                    Model status: {session.model_status.ready ? "Ready" : "Bootstrapping"}
                  </p>
                  <p className="mt-1 text-xs text-slate-500">
                    {session.model_status.classes.length > 0
                      ? session.model_status.classes.join(", ")
                      : "No classes loaded yet"}
                  </p>
                </div>
              </div>

              <div className="mt-5">
                <DrawingCanvas
                  ref={canvasRef}
                  width={440}
                  height={440}
                  className="w-full"
                  onStrokePoint={handleStrokePoint}
                />

                <div className="mt-5 grid grid-cols-1 gap-3 sm:grid-cols-3">
                  <button
                    type="button"
                    onClick={handleGuess}
                    disabled={isPredicting}
                    className="rounded-2xl bg-indigo-600 px-4 py-3 text-sm font-semibold text-white transition hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-70"
                  >
                    {isPredicting ? "Guessing..." : "Guess"}
                  </button>
                  <button
                    type="button"
                    onClick={() => setIsTeachModalOpen(true)}
                    className="rounded-2xl border border-indigo-200 bg-indigo-50 px-4 py-3 text-sm font-semibold text-indigo-700 transition hover:bg-indigo-100"
                  >
                    Teach New Object
                  </button>
                  <button
                    type="button"
                    onClick={handleMultiplayerBattle}
                    className="rounded-2xl border border-slate-300 bg-white px-4 py-3 text-sm font-semibold text-slate-700 transition hover:bg-slate-50"
                  >
                    Multiplayer Battle
                  </button>
                </div>

                <div className="mt-4 rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-xs text-slate-500">
                  Draw with mouse or touch. Every stroke is broadcast to the arena room.
                </div>
              </div>
            </article>

            <div className="grid gap-6 md:grid-cols-2">
              <article className="rounded-[2rem] border border-slate-200 bg-white p-6 shadow-[0_20px_80px_rgba(15,23,42,0.08)]">
                <p className="text-sm font-semibold uppercase tracking-[0.2em] text-indigo-700">
                  Player Profile
                </p>
                <div className="mt-4 space-y-4 text-sm text-slate-600">
                  <div className="flex items-center justify-between">
                    <span>Predictions</span>
                    <span className="font-semibold text-slate-950">{session.user.predictions_count}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>Training sessions</span>
                    <span className="font-semibold text-slate-950">{session.user.training_count}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>Wiki lookups</span>
                    <span className="font-semibold text-slate-950">{session.user.wiki_lookups}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>Wins / Losses</span>
                    <span className="font-semibold text-slate-950">
                      {session.user.wins} / {session.user.losses}
                    </span>
                  </div>
                </div>
              </article>

              <article className="rounded-[2rem] border border-slate-200 bg-white p-6 shadow-[0_20px_80px_rgba(15,23,42,0.08)]">
                <p className="text-sm font-semibold uppercase tracking-[0.2em] text-indigo-700">
                  Model Library
                </p>
                <div className="mt-4 space-y-3">
                  {session.model_status.classes.length > 0 ? (
                    session.model_status.classes.map((cls) => (
                      <div key={cls} className="flex items-center justify-between rounded-xl border border-slate-200 bg-slate-50 px-4 py-2">
                        <span className="font-medium text-slate-800">{cls}</span>
                        <div className="flex gap-2">
                          <button
                            type="button"
                            onClick={() => void handleViewSamples(cls)}
                            className="rounded-lg bg-indigo-50 px-3 py-1 text-xs font-semibold text-indigo-700 transition hover:bg-indigo-100"
                          >
                            View
                          </button>
                          <button
                            type="button"
                            onClick={() => void handleRemoveClass(cls)}
                            className="rounded-lg bg-rose-50 px-3 py-1 text-xs font-semibold text-rose-700 transition hover:bg-rose-100"
                          >
                            Remove
                          </button>
                        </div>
                      </div>
                    ))
                  ) : (
                    <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4 text-sm text-slate-600">
                      The model is ready to learn once enough classes are collected.
                    </div>
                  )}
                </div>
              </article>
            </div>
          </section>

          <aside className="space-y-6">
            <article className="rounded-[2rem] border border-slate-200 bg-white p-6 shadow-[0_20px_80px_rgba(15,23,42,0.08)]">
              <p className="text-sm font-semibold uppercase tracking-[0.2em] text-indigo-700">
                AI Results
              </p>
              <div className="mt-4 space-y-4">
                <div className="rounded-2xl bg-slate-50 p-4">
                  <p className="text-xs font-medium uppercase tracking-[0.16em] text-slate-500">
                    Prediction
                  </p>
                  <p className="mt-2 text-2xl font-semibold text-slate-950">{prediction}</p>
                </div>
                <div className="rounded-2xl bg-slate-50 p-4">
                  <p className="text-xs font-medium uppercase tracking-[0.16em] text-slate-500">
                    Confidence
                  </p>
                  <p className="mt-2 text-2xl font-semibold text-slate-950">
                    {confidence === null ? "-" : `${(confidence * 100).toFixed(1)}%`}
                  </p>
                </div>
                <div className="rounded-2xl border border-indigo-100 bg-indigo-50/70 p-4 text-sm leading-7 text-slate-700">
                  {definition}
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <button
                    type="button"
                    disabled={!lastGuessId}
                    onClick={() => void handleGuessFeedback(true)}
                    className="rounded-2xl border border-emerald-200 bg-emerald-50 px-4 py-2 text-sm font-semibold text-emerald-700 transition hover:bg-emerald-100 disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    Mark Correct
                  </button>
                  <button
                    type="button"
                    disabled={!lastGuessId}
                    onClick={() => void handleGuessFeedback(false)}
                    className="rounded-2xl border border-rose-200 bg-rose-50 px-4 py-2 text-sm font-semibold text-rose-700 transition hover:bg-rose-100 disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    Mark Wrong
                  </button>
                </div>
              </div>
            </article>

            <article className="flex min-h-80 flex-col rounded-[2rem] border border-slate-200 bg-white p-6 shadow-[0_20px_80px_rgba(15,23,42,0.08)]">
              <p className="text-sm font-semibold uppercase tracking-[0.2em] text-indigo-700">
                Multiplayer Chat
              </p>
              <div className="mt-4 flex-1 space-y-3 overflow-y-auto rounded-2xl border border-slate-200 bg-slate-50 p-4">
                {messages.length === 0 ? (
                  <p className="text-sm text-slate-500">
                    No battle messages yet. Join the room to start the duel.
                  </p>
                ) : (
                  messages.map((message, index) => (
                    <div key={`${message}-${index}`} className="rounded-2xl bg-white px-4 py-3 text-sm text-slate-700 shadow-sm">
                      {message}
                    </div>
                  ))
                )}
              </div>
            </article>

            <article className="rounded-[2rem] border border-slate-200 bg-white p-6 shadow-[0_20px_80px_rgba(15,23,42,0.08)]">
              <p className="text-sm font-semibold uppercase tracking-[0.2em] text-indigo-700">
                Recent Activity
              </p>
              <div className="mt-4 space-y-3">
                {session.recent_activity.length === 0 ? (
                  <p className="text-sm text-slate-500">Activity will appear here as you play.</p>
                ) : (
                  session.recent_activity.map((activity) => (
                    <div key={`${activity.kind}-${activity.created_at}`} className="rounded-2xl bg-slate-50 p-4 text-sm text-slate-600">
                      <p className="font-semibold text-slate-900">{formatActivity(activity.kind)}</p>
                      <p className="mt-1">{activity.detail}</p>
                    </div>
                  ))
                )}
              </div>
            </article>
          </aside>
        </main>

        <section className="grid gap-6 lg:grid-cols-2 xl:grid-cols-3">
          <article className="rounded-[2rem] border border-slate-200 bg-white p-6 shadow-[0_20px_80px_rgba(15,23,42,0.08)] lg:col-span-2 xl:col-span-3">
            <p className="text-sm font-semibold uppercase tracking-[0.2em] text-indigo-700">
              QuickDraw Database Trainer
            </p>
            <p className="mt-2 text-sm text-slate-600">
              Choose classes from Google QuickDraw and import real doodle samples to improve model diversity.
            </p>

            <div className="mt-4 flex flex-wrap items-center gap-3">
              <label className="text-sm font-medium text-slate-700">
                Samples / class
                <input
                  type="number"
                  min={10}
                  max={120}
                  value={quickdrawSamples}
                  onChange={(event) => setQuickdrawSamples(Number(event.target.value) || 40)}
                  className="ml-2 w-24 rounded-lg border border-slate-300 px-2 py-1 text-sm"
                />
              </label>
              <button
                type="button"
                disabled={quickdrawLoading || selectedQuickdrawClasses.length === 0}
                onClick={() => void handleQuickdrawTrain()}
                className="rounded-2xl bg-indigo-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {quickdrawLoading ? "Importing..." : "Train From QuickDraw"}
              </button>
              <p className="text-xs text-slate-500">
                Selected: {selectedQuickdrawClasses.length} classes
              </p>
            </div>

            <div className="mt-4 grid max-h-52 grid-cols-2 gap-2 overflow-y-auto rounded-2xl border border-slate-200 bg-slate-50 p-3 sm:grid-cols-3 lg:grid-cols-4">
              {quickdrawClasses.length === 0 ? (
                <p className="col-span-full text-sm text-slate-500">Loading QuickDraw class list...</p>
              ) : (
                quickdrawClasses.map((className) => {
                  const selected = selectedQuickdrawClasses.includes(className);
                  return (
                    <button
                      key={className}
                      type="button"
                      onClick={() => handleToggleQuickdrawClass(className)}
                      className={`rounded-xl border px-3 py-2 text-left text-xs font-medium transition ${
                        selected
                          ? "border-indigo-300 bg-indigo-100 text-indigo-800"
                          : "border-slate-200 bg-white text-slate-700 hover:bg-slate-100"
                      }`}
                    >
                      {className}
                    </button>
                  );
                })
              )}
            </div>
          </article>

          <article className="rounded-[2rem] border border-slate-200 bg-white p-6 shadow-[0_20px_80px_rgba(15,23,42,0.08)]">
            <p className="text-sm font-semibold uppercase tracking-[0.2em] text-indigo-700">
              Missions
            </p>
            <div className="mt-4 space-y-4">
              {session.missions.map((mission) => {
                const percent = Math.max(0, Math.min(100, (mission.progress / mission.target) * 100));
                return (
                  <div key={mission.id} className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <p className="font-semibold text-slate-950">{mission.title}</p>
                        <p className="mt-1 text-sm text-slate-600">{mission.description}</p>
                      </div>
                      <span className="rounded-full bg-white px-3 py-1 text-xs font-semibold text-indigo-700">
                        {mission.reward}
                      </span>
                    </div>
                    <div className="mt-3 h-2 rounded-full bg-slate-200">
                      <div
                        className="h-2 rounded-full bg-indigo-600 transition-all"
                        style={{ width: `${percent}%` }}
                      />
                    </div>
                    <p className="mt-2 text-xs text-slate-500">
                      {mission.progress} / {mission.target}
                    </p>
                  </div>
                );
              })}
            </div>
          </article>

          <article className="rounded-[2rem] border border-slate-200 bg-white p-6 shadow-[0_20px_80px_rgba(15,23,42,0.08)]">
            <p className="text-sm font-semibold uppercase tracking-[0.2em] text-indigo-700">
              Leaderboard
            </p>
            <div className="mt-4 space-y-3">
              {session.leaderboard.map((player, index) => (
                <div key={player.username} className="flex items-center justify-between rounded-2xl bg-slate-50 px-4 py-3">
                  <div>
                    <p className="font-semibold text-slate-950">
                      #{index + 1} {player.display_name}
                    </p>
                    <p className="text-xs text-slate-500">@{player.username}</p>
                  </div>
                  <div className="text-right text-sm text-slate-600">
                    <p className="font-semibold text-slate-950">Lv {player.level}</p>
                    <p>{player.xp} XP · {player.coins} coins</p>
                  </div>
                </div>
              ))}
            </div>
          </article>

          <article className="rounded-[2rem] border border-slate-200 bg-white p-6 shadow-[0_20px_80px_rgba(15,23,42,0.08)] lg:col-span-2 xl:col-span-1">
            <p className="text-sm font-semibold uppercase tracking-[0.2em] text-indigo-700">
              Achievements
            </p>
            <div className="mt-4 grid gap-3">
              {session.achievements.map((achievement) => (
                <div
                  key={achievement.id}
                  className={`rounded-2xl border p-4 ${
                    achievement.earned
                      ? "border-indigo-200 bg-indigo-50"
                      : "border-slate-200 bg-slate-50"
                  }`}
                >
                  <p className="font-semibold text-slate-950">{achievement.title}</p>
                  <p className="mt-1 text-sm text-slate-600">{achievement.description}</p>
                  <p className={`mt-2 text-xs font-semibold ${achievement.earned ? "text-indigo-700" : "text-slate-500"}`}>
                    {achievement.earned ? "Earned" : "Locked"}
                  </p>
                </div>
              ))}
            </div>
          </article>

          <article className="rounded-[2rem] border border-slate-200 bg-white p-6 shadow-[0_20px_80px_rgba(15,23,42,0.08)] lg:col-span-2 xl:col-span-1">
            <p className="text-sm font-semibold uppercase tracking-[0.2em] text-indigo-700">
              Correct Guess History
            </p>
            <div className="mt-4 space-y-3">
              {session.correct_guesses.length === 0 ? (
                <p className="text-sm text-slate-500">
                  Mark predictions as correct to build your verified guess list.
                </p>
              ) : (
                session.correct_guesses.map((item) => (
                  <div key={item.guess_id} className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3">
                    <p className="font-semibold text-slate-950">{item.prediction}</p>
                    <p className="mt-1 text-xs text-slate-500">
                      Confidence {(item.confidence * 100).toFixed(1)}%
                    </p>
                  </div>
                ))
              )}
            </div>
          </article>
        </section>

        {toast ? (
          <div className="fixed bottom-5 right-5 rounded-2xl bg-indigo-600 px-4 py-3 text-sm font-semibold text-white shadow-[0_20px_80px_rgba(79,70,229,0.35)]">
            {toast}
          </div>
        ) : null}

        <TeachModal
          isOpen={isTeachModalOpen}
          onClose={() => setIsTeachModalOpen(false)}
          onTrained={handleModalTrained}
        />

        {isSamplesModalOpen && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm">
            <div className="w-full max-w-4xl rounded-3xl bg-white p-8 shadow-2xl">
              <div className="flex items-center justify-between border-b pb-4">
                <h2 className="text-2xl font-semibold text-slate-900">
                  Training Samples for &ldquo;{viewingClass}&rdquo;
                </h2>
                <button
                  type="button"
                  onClick={() => setIsSamplesModalOpen(false)}
                  className="rounded-full p-2 hover:bg-slate-100"
                >
                  ✕
                </button>
              </div>

              <div className="mt-6">
                {samplesLoading ? (
                  <div className="flex h-64 items-center justify-center">
                    <p className="animate-pulse text-slate-500">Retrieving doodles...</p>
                  </div>
                ) : (
                  <div className="grid grid-cols-1 gap-4 sm:grid-cols-5">
                    {classSamples.map((pixelData, i) => (
                      <div key={i} className="flex flex-col items-center gap-2">
                        <SmallCanvas pixels={pixelData} />
                        <span className="text-[10px] uppercase tracking-wider text-slate-400">
                          Sample {i + 1}
                        </span>
                      </div>
                    ))}
                    {classSamples.length === 0 && (
                      <p className="col-span-full py-12 text-center text-slate-500">
                        No stored drawings found for this class.
                      </p>
                    )}
                  </div>
                )}
              </div>

              <div className="mt-8 flex justify-end">
                <button
                  type="button"
                  onClick={() => setIsSamplesModalOpen(false)}
                  className="rounded-2xl bg-indigo-600 px-6 py-2 text-sm font-semibold text-white transition hover:bg-indigo-500"
                >
                  Close
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  ) : authScreen;

  if (authLoading) {
    return loadingScreen;
  }

  return session ? dashboardScreen : authScreen;
}

function SmallCanvas({ pixels }: { pixels: number[][] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const size = pixels.length;
    canvas.width = size;
    canvas.height = size;

    const imageData = ctx.createImageData(size, size);
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const val = pixels[y][x];
        const idx = (y * size + x) * 4;
        imageData.data[idx] = val;     // R
        imageData.data[idx + 1] = val; // G
        imageData.data[idx + 2] = val; // B
        imageData.data[idx + 3] = 255; // A
      }
    }
    ctx.putImageData(imageData, 0, 0);
  }, [pixels]);

  return (
    <canvas
      ref={canvasRef}
      className="h-32 w-32 rounded-xl border border-slate-200 bg-white shadow-sm"
      style={{ imageRendering: "pixelated" }}
    />
  );
}