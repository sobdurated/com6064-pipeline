import argparse
import json
import os
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS


VALID_STEPS = [
    "all", "fetch", "preprocess", "sentiment",
    "sentiment_llm", "geospatial", "sentiment_aggregation",
]


class PipelineRunner:
    def __init__(self, pipeline_dir: str, socketio_instance: SocketIO):
        self.pipeline_dir = pipeline_dir
        self.socketio = socketio_instance
        self.runs = []
        self.current_run = None
        self.process = None
        self.lock = threading.Lock()

    def start_run(self, step, input_data=None):
        with self.lock:
            if self.current_run and self.current_run["status"] == "running":
                return None, "A pipeline step is already running"

            run_id = str(uuid.uuid4())[:8]
            run = {
                "id": run_id,
                "step": step,
                "input_data": input_data,
                "status": "running",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "finished_at": None,
                "return_code": None,
                "logs": [],
            }
            self.current_run = run
            self.runs.append(run)

        thread = threading.Thread(target=self._execute, args=(run,), daemon=True)
        thread.start()
        return run, None

    def _execute(self, run):
        step = run["step"]
        input_data = run.get("input_data")

        if step == "all":
            cmd = [sys.executable, "pipeline.py"]
        else:
            cmd = [sys.executable, "pipeline.py", "--step", step]

        if input_data:
            cmd.extend(["--input", json.dumps(input_data)])

        def _log(text):
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "text": text,
                "run_id": run["id"],
            }
            run["logs"].append(entry)
            self.socketio.emit("log", entry)

        _log(f">>> Starting: {' '.join(cmd)}")
        _log(f">>> Working dir: {self.pipeline_dir}")

        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=self.pipeline_dir,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )

            for line in iter(self.process.stdout.readline, ""):
                _log(line.rstrip("\n\r"))

            self.process.wait()
            run["return_code"] = self.process.returncode
            run["status"] = "completed" if self.process.returncode == 0 else "error"
            _log(f">>> Finished with exit code {self.process.returncode}")

        except Exception as e:
            run["status"] = "error"
            _log(f">>> [SERVER ERROR] {e}")

        finally:
            run["finished_at"] = datetime.now(timezone.utc).isoformat()
            self.process = None
            self.socketio.emit("run_finished", {
                "id": run["id"],
                "step": run["step"],
                "status": run["status"],
                "return_code": run["return_code"],
                "finished_at": run["finished_at"],
            })

    def stop_run(self):
        if self.process:
            self.process.terminate()
            return True
        return False

    def get_status(self):
        if not self.current_run:
            return {"status": "idle"}
        return {
            "id": self.current_run["id"],
            "step": self.current_run["step"],
            "status": self.current_run["status"],
            "started_at": self.current_run["started_at"],
            "finished_at": self.current_run["finished_at"],
            "log_count": len(self.current_run["logs"]),
        }


def create_app(pipeline_dir: str) -> tuple:
    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": "*"}})
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
    runner = PipelineRunner(pipeline_dir, socketio)

    @app.route("/api/run", methods=["POST"])
    def trigger_run():
        data = request.json or {}
        step = data.get("step", "all")
        input_data = data.get("input", None)
        if step not in VALID_STEPS:
            return jsonify({"error": f"Invalid step. Valid: {VALID_STEPS}"}), 400

        run, error = runner.start_run(step, input_data=input_data)
        if error:
            return jsonify({"error": error}), 409

        return jsonify({
            "message": f"Started: {step}",
            "id": run["id"],
            "step": run["step"],
            "started_at": run["started_at"],
        })

    @app.route("/api/status")
    def get_status():
        return jsonify(runner.get_status())

    @app.route("/api/logs")
    def get_logs():
        run_id = request.args.get("run_id")
        if run_id:
            for r in runner.runs:
                if r["id"] == run_id:
                    return jsonify(r["logs"])
            return jsonify([])
        if runner.current_run:
            return jsonify(runner.current_run["logs"])
        return jsonify([])

    @app.route("/api/runs")
    def get_runs():
        return jsonify([
            {
                "id": r["id"],
                "step": r["step"],
                "status": r["status"],
                "started_at": r["started_at"],
                "finished_at": r["finished_at"],
                "log_count": len(r["logs"]),
            }
            for r in runner.runs
        ])

    @app.route("/api/stop", methods=["POST"])
    def stop_run():
        if runner.stop_run():
            return jsonify({"message": "Stop signal sent"})
        return jsonify({"error": "No running process"}), 404

    @app.route("/api/steps")
    def get_steps():
        return jsonify(VALID_STEPS)

    @socketio.on("connect")
    def on_connect():
        emit("status", runner.get_status())
        if runner.current_run:
            emit("logs_replay", runner.current_run["logs"])

    @socketio.on("trigger_run")
    def ws_trigger_run(data):
        step = (data or {}).get("step", "all")
        input_data = (data or {}).get("input", None)
        if step not in VALID_STEPS:
            emit("error", {"message": f"Invalid step: {step}"})
            return

        run, error = runner.start_run(step, input_data=input_data)
        if error:
            emit("error", {"message": error})
            return

        socketio.emit("run_started", {
            "id": run["id"],
            "step": run["step"],
            "started_at": run["started_at"],
        })

    return app, socketio, runner


def start_server(pipeline_dir: str, port: int = 5000, ngrok_token: str = None, host: str = "127.0.0.1", background: bool = False):
    app, socketio, runner = create_app(pipeline_dir)

    public_url = f"http://{host}:{port}"

    if ngrok_token:
        from pyngrok import ngrok
        ngrok.set_auth_token(ngrok_token)
        tunnel = ngrok.connect(port, bind_tls=True)
        public_url = tunnel.public_url

    print("\n" + "=" * 60)
    print(f"  Pipeline Server")
    print(f"  URL: {public_url}")
    print(f"  Local Address: http://{host}:{port}")
    print(f"  Pipeline dir: {pipeline_dir}")
    print("=" * 60)
    print(f"""
REST:
  POST {public_url}/api/run   → {{"step": "sentiment", "input": {{...}}}}
  GET  {public_url}/api/status
  GET  {public_url}/api/logs
  GET  {public_url}/api/runs
  POST {public_url}/api/stop

WebSocket: connect to {public_url}
  Events: log, logs_replay, status, run_started, run_finished
""")

    if background:
        thread = threading.Thread(
            target=socketio.run,
            args=(app,),
            kwargs={"host": host, "port": port, "debug": False, "allow_unsafe_werkzeug": True},
            daemon=True
        )
        thread.start()
        print("Server running in background thread.")
        return thread
    else:
        socketio.run(app, host=host, port=port, debug=False, allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="Pipeline WebSocket server")
    parser.add_argument("--pipeline-dir", default=str(Path(__file__).parent),
                        help="Directory containing pipeline scripts (default: same dir as this file)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Host to bind server to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "5000")),
                        help="Server port (default: 5000)")
    parser.add_argument("--ngrok-token", default=os.getenv("NGROK_AUTH_TOKEN"),
                        help="ngrok auth token for tunneling (optional)")
    parser.add_argument("--background", action="store_true",
                        help="Run server in a background thread")
    args = parser.parse_args()

    start_server(
        pipeline_dir=args.pipeline_dir,
        port=args.port,
        ngrok_token=args.ngrok_token,
        host=args.host,
        background=args.background,
    )