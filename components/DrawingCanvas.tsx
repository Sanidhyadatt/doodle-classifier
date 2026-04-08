"use client";

import {
  type MouseEvent,
  type TouchEvent,
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react";

export type DrawingCanvasRef = {
  clearCanvas: () => void;
  getGrayscalePixels: () => number[][];
  drawRemotePoint: (x: number, y: number) => void;
};

type DrawingCanvasProps = {
  width?: number;
  height?: number;
  className?: string;
  onStrokePoint?: (x: number, y: number) => void;
};

const DrawingCanvas = forwardRef<DrawingCanvasRef, DrawingCanvasProps>(
  ({ width = 280, height = 280, className = "", onStrokePoint }, ref) => {
    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const [isDrawing, setIsDrawing] = useState(false);
    const lastRemotePointRef = useRef<{ x: number; y: number } | null>(null);

    const initializeCanvas = useCallback(() => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = "#000000";
      ctx.lineWidth = 6;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
    }, []);

    useEffect(() => {
      initializeCanvas();
    }, [initializeCanvas]);

    const getPointFromMouse = (event: MouseEvent<HTMLCanvasElement>) => {
      const canvas = event.currentTarget;
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      return {
        x: (event.clientX - rect.left) * scaleX,
        y: (event.clientY - rect.top) * scaleY,
      };
    };

    const getPointFromTouch = (event: TouchEvent<HTMLCanvasElement>) => {
      const canvas = event.currentTarget;
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      const touch = event.touches[0] ?? event.changedTouches[0];
      if (!touch) return { x: 0, y: 0 };
      return {
        x: (touch.clientX - rect.left) * scaleX,
        y: (touch.clientY - rect.top) * scaleY,
      };
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
      onStrokePoint?.(x, y);
    };

    const endStroke = () => {
      setIsDrawing(false);
    };

    const clearCanvas = useCallback(() => {
      lastRemotePointRef.current = null;
      initializeCanvas();
    }, [initializeCanvas]);

    const drawRemotePoint = useCallback((x: number, y: number) => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const previous = lastRemotePointRef.current;
      if (!previous) {
        ctx.beginPath();
        ctx.moveTo(x, y);
      } else {
        ctx.beginPath();
        ctx.moveTo(previous.x, previous.y);
        ctx.lineTo(x, y);
        ctx.stroke();
      }

      lastRemotePointRef.current = { x, y };
    }, []);

    const getGrayscalePixels = useCallback((): number[][] => {
      const canvas = canvasRef.current;
      if (!canvas) return [];

      const ctx = canvas.getContext("2d");
      if (!ctx) return [];

      const { data, width: w, height: h } = ctx.getImageData(
        0,
        0,
        canvas.width,
        canvas.height,
      );

      const rows: number[][] = [];
      for (let y = 0; y < h; y += 1) {
        const row: number[] = [];
        for (let x = 0; x < w; x += 1) {
          const idx = (y * w + x) * 4;
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

    useImperativeHandle(ref, () => ({
      clearCanvas,
      getGrayscalePixels,
      drawRemotePoint,
    }));

    return (
      <div className={`inline-flex flex-col gap-3 ${className}`.trim()}>
        <canvas
          ref={canvasRef}
          width={width}
          height={height}
          className="touch-none rounded-lg border border-zinc-300 bg-white"
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

        <button
          type="button"
          onClick={clearCanvas}
          className="self-start rounded-md bg-zinc-900 px-3 py-1.5 text-sm font-medium text-white hover:bg-zinc-700"
        >
          Clear
        </button>
      </div>
    );
  },
);

DrawingCanvas.displayName = "DrawingCanvas";

export default DrawingCanvas;
