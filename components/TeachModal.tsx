"use client";

import axios from "axios";
import {
  type MouseEvent,
  type TouchEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

type TeachModalProps = {
  isOpen: boolean;
  onClose: () => void;
  onTrained?: (className: string) => void;
};

const TOTAL_DRAWINGS = 5;

export default function TeachModal({ isOpen, onClose, onTrained }: TeachModalProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [className, setClassName] = useState("");
  const [samples, setSamples] = useState<number[][][]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [toastMessage, setToastMessage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const progress = samples.length;
  const progressPercent = useMemo(
    () => Math.round((progress / TOTAL_DRAWINGS) * 100),
    [progress],
  );

  const initializeCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "#000000";
    ctx.lineWidth = 5;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
  }, []);

  useEffect(() => {
    if (!isOpen) return;
    initializeCanvas();
    setSamples([]);
    setErrorMessage(null);
    setToastMessage(null);
  }, [initializeCanvas, isOpen]);

  const clearCanvas = useCallback(() => {
    initializeCanvas();
  }, [initializeCanvas]);

  const getPointFromMouse = (event: MouseEvent<HTMLCanvasElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    return { x: event.clientX - rect.left, y: event.clientY - rect.top };
  };

  const getPointFromTouch = (event: TouchEvent<HTMLCanvasElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const touch = event.touches[0] ?? event.changedTouches[0];
    return { x: touch.clientX - rect.left, y: touch.clientY - rect.top };
  };

  const startStroke = (x: number, y: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.beginPath();
    ctx.moveTo(x, y);
    setIsDrawing(true);
  };

  const continueStroke = (x: number, y: number) => {
    if (!isDrawing) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.lineTo(x, y);
    ctx.stroke();
  };

  const endStroke = () => {
    setIsDrawing(false);
  };

  const getGrayscalePixels = useCallback((): number[][] => {
    const canvas = canvasRef.current;
    if (!canvas) return [];

    const ctx = canvas.getContext("2d");
    if (!ctx) return [];

    const { data, width, height } = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const rows: number[][] = [];

    for (let y = 0; y < height; y += 1) {
      const row: number[] = [];
      for (let x = 0; x < width; x += 1) {
        const idx = (y * width + x) * 4;
        const r = data[idx];
        const g = data[idx + 1];
        const b = data[idx + 2];
        const gray = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
        row.push(gray);
      }
      rows.push(row);
    }

    return rows;
  }, []);

  const submitTraining = useCallback(
    async (nextSamples: number[][][]) => {
      const cleanedName = className.trim();
      if (!cleanedName || nextSamples.length !== TOTAL_DRAWINGS) return;

      setIsSubmitting(true);
      setErrorMessage(null);

      try {
        await axios.post(
          "http://localhost:8000/train_class",
          {
          class_name: cleanedName,
          pixel_data: nextSamples,
          },
          { withCredentials: true },
        );

        setToastMessage(`Successfully trained class: ${cleanedName}`);
        onTrained?.(cleanedName);
      } catch {
        setErrorMessage("Training failed. Please check backend connection and try again.");
      } finally {
        setIsSubmitting(false);
      }
    },
    [className, onTrained],
  );

  const handleSubmitDrawing = async () => {
    if (isSubmitting) return;

    const pixels = getGrayscalePixels();
    if (!pixels.length) return;

    const nextSamples = [...samples, pixels];
    setSamples(nextSamples);
    clearCanvas();

    if (nextSamples.length === TOTAL_DRAWINGS) {
      await submitTraining(nextSamples);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 p-4">
      <div className="w-full max-w-lg rounded-2xl border border-slate-200 bg-white p-6 shadow-xl">
        <div className="flex items-start justify-between">
          <div>
            <h2 className="text-xl font-semibold text-slate-900">Teach New Object</h2>
            <p className="mt-1 text-sm text-slate-500">
              Enter a name and draw it 5 times for the model.
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-md px-2 py-1 text-slate-500 hover:bg-slate-100"
          >
            Close
          </button>
        </div>

        <div className="mt-4 space-y-4">
          <label className="block text-sm font-medium text-slate-700">
            New Object Name
            <input
              type="text"
              placeholder="e.g., Samosa"
              value={className}
              onChange={(event) => setClassName(event.target.value)}
              className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm text-slate-800 outline-none ring-indigo-200 focus:ring"
            />
          </label>

          <canvas
            ref={canvasRef}
            width={220}
            height={220}
            className="mx-auto touch-none rounded-lg border border-slate-300 bg-white"
            onMouseDown={(event) => {
              const point = getPointFromMouse(event);
              startStroke(point.x, point.y);
            }}
            onMouseMove={(event) => {
              const point = getPointFromMouse(event);
              continueStroke(point.x, point.y);
            }}
            onMouseUp={endStroke}
            onMouseLeave={endStroke}
            onTouchStart={(event) => {
              event.preventDefault();
              const point = getPointFromTouch(event);
              startStroke(point.x, point.y);
            }}
            onTouchMove={(event) => {
              event.preventDefault();
              const point = getPointFromTouch(event);
              continueStroke(point.x, point.y);
            }}
            onTouchEnd={(event) => {
              event.preventDefault();
              endStroke();
            }}
            onTouchCancel={endStroke}
          />

          <div>
            <div className="mb-1 flex items-center justify-between text-xs font-medium text-slate-600">
              <span>Progress</span>
              <span>
                {progress}/{TOTAL_DRAWINGS} drawn
              </span>
            </div>
            <div className="h-2 w-full rounded-full bg-slate-200">
              <div
                className="h-2 rounded-full bg-indigo-600 transition-all"
                style={{ width: `${progressPercent}%` }}
              />
            </div>
          </div>

          {errorMessage ? <p className="text-sm text-red-600">{errorMessage}</p> : null}

          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={handleSubmitDrawing}
              disabled={isSubmitting || !className.trim() || progress >= TOTAL_DRAWINGS}
              className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-70"
            >
              {isSubmitting ? "Training..." : "Submit Drawing"}
            </button>
            <button
              type="button"
              onClick={clearCanvas}
              className="rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50"
            >
              Clear Canvas
            </button>
          </div>
        </div>
      </div>

      {toastMessage ? (
        <div className="fixed bottom-5 right-5 rounded-lg bg-emerald-600 px-4 py-3 text-sm font-semibold text-white shadow-lg">
          {toastMessage}
        </div>
      ) : null}
    </div>
  );
}
