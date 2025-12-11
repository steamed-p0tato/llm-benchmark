#!/usr/bin/env python3
"""
System Benchmark TUI Application
A comprehensive CPU and GPU benchmarking tool using Textual
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer, Center
from textual.widgets import (
    Header, Footer, Button, Static, Label, 
    ProgressBar, DataTable, TabbedContent, 
    TabPane, Rule, LoadingIndicator, RichLog
)
from textual.screen import Screen, ModalScreen
from textual.binding import Binding
from textual import work
from textual.worker import Worker, get_current_worker
from textual.reactive import reactive

from venv_manager import VenvManager
from cpu_benchmark import MandelbrotBenchmark, CPUBenchmarkResult
from gpu_benchmark import LLMBenchmark, GPUBenchmarkResult


# Styles
CSS = """
Screen {
    background: $surface;
}

#main-container {
    width: 100%;
    height: 100%;
    padding: 1 2;
}

#title-box {
    height: auto;
    width: 100%;
    content-align: center middle;
    padding: 1;
    margin-bottom: 1;
    border: solid $primary;
    background: $boost;
}

#title {
    text-align: center;
    text-style: bold;
    color: $text;
}

#subtitle {
    text-align: center;
    color: $text-muted;
}

#status-panel {
    height: auto;
    width: 100%;
    margin-bottom: 1;
    padding: 1;
    border: solid $secondary;
    background: $surface-darken-1;
}

.status-row {
    height: 3;
    width: 100%;
}

.status-label {
    width: 20;
    color: $text-muted;
}

.status-value {
    width: 1fr;
    color: $text;
}

.status-ok {
    color: $success;
}

.status-warn {
    color: $warning;
}

.status-error {
    color: $error;
}

#button-panel {
    height: auto;
    width: 100%;
    layout: horizontal;
    margin-bottom: 1;
}

#button-panel Button {
    margin: 0 1;
}

.action-button {
    min-width: 24;
}

#venv-button {
    background: $primary;
}

#cpu-button {
    background: $success-darken-2;
}

#gpu-button {
    background: $warning-darken-2;
}

#results-button {
    background: $accent;
}

#log-container {
    height: 1fr;
    width: 100%;
    border: solid $primary-darken-2;
    background: $surface-darken-2;
}

#benchmark-log {
    height: 100%;
    width: 100%;
    scrollbar-gutter: stable;
}

#progress-container {
    height: auto;
    width: 100%;
    padding: 1;
    dock: bottom;
    background: $surface-darken-1;
}

#progress-label {
    text-align: center;
    margin-bottom: 1;
}

#progress-bar {
    width: 100%;
}

ModalScreen {
    align: center middle;
}

#results-modal {
    width: 90%;
    height: 80%;
    border: thick $primary;
    background: $surface;
    padding: 1 2;
}

#results-content {
    width: 100%;
    height: 1fr;
    overflow-y: auto;
}

#results-title {
    text-align: center;
    text-style: bold;
    color: $primary;
    margin-bottom: 1;
}

.results-section {
    margin: 1 0;
    padding: 1;
    border: solid $secondary;
}

.results-heading {
    text-style: bold;
    color: $secondary;
    margin-bottom: 1;
}

DataTable {
    height: auto;
    max-height: 20;
}

#close-results {
    margin-top: 1;
    width: 100%;
}

#setup-modal {
    width: 70%;
    height: 70%;
    border: thick $primary;
    background: $surface;
    padding: 2;
}

#setup-log {
    height: 1fr;
    width: 100%;
    background: $surface-darken-2;
    margin: 1 0;
}

#setup-progress {
    width: 100%;
    margin: 1 0;
}

#setup-buttons {
    height: auto;
    layout: horizontal;
    align: center middle;
}

.warning-text {
    color: $warning;
    text-style: italic;
}

#confirm-modal {
    width: 50%;
    height: auto;
    min-height: 15;
    border: thick $warning;
    background: $surface;
    padding: 2;
}

#confirm-title {
    text-align: center;
    text-style: bold;
    color: $warning;
    margin-bottom: 1;
}

#confirm-message {
    text-align: center;
    margin-bottom: 2;
}

#confirm-buttons {
    layout: horizontal;
    align: center middle;
    height: auto;
}

#confirm-buttons Button {
    margin: 0 1;
}

.hidden {
    display: none;
}
"""


class ResultsScreen(ModalScreen):
    """Modal screen for displaying benchmark results"""
    
    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
    ]
    
    def __init__(
        self, 
        cpu_result: Optional[CPUBenchmarkResult] = None,
        gpu_result: Optional[GPUBenchmarkResult] = None
    ):
        super().__init__()
        self.cpu_result = cpu_result
        self.gpu_result = gpu_result
    
    def compose(self) -> ComposeResult:
        with Container(id="results-modal"):
            yield Static("📊 BENCHMARK RESULTS", id="results-title")
            yield Rule()
            
            with ScrollableContainer(id="results-content"):
                # CPU Results
                if self.cpu_result:
                    with Container(classes="results-section"):
                        yield Static("🖥️ CPU BENCHMARK", classes="results-heading")
                        
                        cpu_table = DataTable(id="cpu-table")
                        cpu_table.add_columns("Metric", "Value")
                        cpu_table.add_rows([
                            ("CPU Model", str(self.cpu_result.cpu_info.get('model_name', 'N/A'))),
                            ("Cores", str(self.cpu_result.core_count)),
                            ("Resolution", f"{self.cpu_result.width}x{self.cpu_result.height}"),
                            ("Max Iterations", str(self.cpu_result.max_iterations)),
                            ("Single-Core Time", f"{self.cpu_result.single_core_time:.2f}s"),
                            ("Multi-Core Time", f"{self.cpu_result.multi_core_time:.2f}s"),
                            ("Pixels/Second", f"{self.cpu_result.pixels_per_second:,.0f}"),
                            ("Scaling Efficiency", f"{self.cpu_result.scaling_efficiency:.1f}%"),
                            ("★ SCORE", f"{self.cpu_result.score:,.0f}"),
                        ])
                        yield cpu_table
                
                # GPU Results
                if self.gpu_result:
                    with Container(classes="results-section"):
                        yield Static("🎮 GPU BENCHMARK", classes="results-heading")
                        
                        if self.gpu_result.success:
                            gpu_table = DataTable(id="gpu-table")
                            gpu_table.add_columns("Metric", "Value")
                            gpu_table.add_rows([
                                ("Model", self.gpu_result.model_name.split("/")[-1] if self.gpu_result.model_name else "N/A"),
                                ("GPU", self.gpu_result.gpu_info.get('name', 'N/A')),
                                ("Backend", self.gpu_result.backend.upper()),
                                ("Prompt Tokens", str(self.gpu_result.prompt_tokens)),
                                ("Generated Tokens", str(self.gpu_result.generated_tokens)),
                                ("Total Time", f"{self.gpu_result.total_time:.2f}s"),
                                ("★ Tokens/Second", f"{self.gpu_result.tokens_per_second:.2f}"),
                                ("Memory Used", f"{self.gpu_result.memory_used_gb:.2f} GB"),
                                ("Memory Total", f"{self.gpu_result.memory_total_gb:.2f} GB"),
                            ])
                            yield gpu_table
                            
                            yield Static("Response Preview:", classes="results-heading")
                            response_preview = self.gpu_result.response[:500] + "..." if len(self.gpu_result.response) > 500 else self.gpu_result.response
                            yield Static(response_preview)
                        else:
                            yield Static(f"❌ Benchmark Failed: {self.gpu_result.error_message}", classes="status-error")
                
                if not self.cpu_result and not self.gpu_result:
                    yield Static("No benchmark results available. Run a benchmark first!", classes="warning-text")
            
            yield Button("Close", id="close-results", variant="primary")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "close-results":
            self.dismiss()


class SetupScreen(ModalScreen):
    """Modal screen for venv setup"""
    
    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]
    
    setup_complete = reactive(False)
    setup_running = reactive(False)
    
    def __init__(self, venv_manager: VenvManager):
        super().__init__()
        self.venv_manager = venv_manager
    
    def compose(self) -> ComposeResult:
        with Container(id="setup-modal"):
            yield Static("🔧 Virtual Environment Setup", id="results-title")
            yield Rule()
            yield RichLog(id="setup-log", highlight=True, markup=True)
            yield ProgressBar(id="setup-progress", show_eta=False)
            
            with Horizontal(id="setup-buttons"):
                yield Button("Start Setup", id="start-setup", variant="primary")
                yield Button("Cancel", id="cancel-setup", variant="error")
    
    def on_mount(self) -> None:
        self.query_one("#setup-progress", ProgressBar).update(total=100, progress=0)
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start-setup" and not self.setup_running:
            self.run_setup()
        elif event.button.id == "cancel-setup":
            self.dismiss(False)
    
    @work(exclusive=True, thread=True)
    def run_setup(self) -> None:
        """Run the venv setup in a worker thread"""
        self.setup_running = True
        
        # Get widgets
        log_widget = self.query_one("#setup-log", RichLog)
        progress_bar = self.query_one("#setup-progress", ProgressBar)
        start_button = self.query_one("#start-setup", Button)
        cancel_button = self.query_one("#cancel-setup", Button)
        
        # Hide start button - use app.call_from_thread
        self.app.call_from_thread(start_button.add_class, "hidden")
        
        def log_callback(message: str):
            self.app.call_from_thread(log_widget.write, message)
        
        def progress_callback(progress: float):
            self.app.call_from_thread(progress_bar.update, progress=progress * 100)
        
        self.venv_manager.set_callbacks(log_callback, progress_callback)
        
        success = self.venv_manager.setup_complete_environment()
        
        if success:
            # Verify installation
            self.app.call_from_thread(log_widget.write, "")
            self.app.call_from_thread(log_widget.write, "Verifying installation...")
            
            verified = self.venv_manager.verify_installation()
            if verified:
                self.app.call_from_thread(log_widget.write, "✅ Setup completed and verified successfully!")
                self.app.call_from_thread(log_widget.write, "You can now run benchmarks.")
                self.setup_complete = True
            else:
                self.app.call_from_thread(log_widget.write, "⚠️ Setup completed but verification failed.")
                self.app.call_from_thread(log_widget.write, "Try running setup again.")
                self.setup_complete = False
        else:
            self.app.call_from_thread(log_widget.write, "")
            self.app.call_from_thread(log_widget.write, "❌ Setup failed. Please check the logs above.")
        
        self.setup_running = False
        
        # Change cancel button to close
        def update_button():
            cancel_button.label = "Close"
        
        self.app.call_from_thread(update_button)
    
    def action_cancel(self) -> None:
        if not self.setup_running:
            self.dismiss(self.setup_complete)


class ConfirmScreen(ModalScreen):
    """Confirmation dialog"""
    
    def __init__(self, title: str, message: str):
        super().__init__()
        self.dialog_title = title
        self.dialog_message = message
    
    def compose(self) -> ComposeResult:
        with Container(id="confirm-modal"):
            yield Static(self.dialog_title, id="confirm-title")
            yield Static(self.dialog_message, id="confirm-message")
            
            with Horizontal(id="confirm-buttons"):
                yield Button("Yes", id="confirm-yes", variant="success")
                yield Button("No", id="confirm-no", variant="error")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "confirm-yes")


class BenchmarkApp(App):
    """Main Benchmark TUI Application"""
    
    CSS = CSS
    TITLE = "System Benchmark"
    SUB_TITLE = "CPU & GPU Performance Testing"
    
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("v", "check_venv", "Check Venv"),
        Binding("c", "run_cpu", "CPU Bench"),
        Binding("g", "run_gpu", "GPU Bench"),
        Binding("r", "show_results", "Results"),
        Binding("d", "toggle_dark", "Dark Mode"),
    ]
    
    # Reactive attributes
    venv_status = reactive("Checking...")
    gpu_status = reactive("Unknown")
    benchmark_running = reactive(False)
    venv_ready = reactive(False)
    
    def __init__(self):
        super().__init__()
        self.venv_manager = VenvManager()
        self.cpu_result: Optional[CPUBenchmarkResult] = None
        self.gpu_result: Optional[GPUBenchmarkResult] = None
    
    def compose(self) -> ComposeResult:
        yield Header()
        
        with Container(id="main-container"):
            # Title
            with Container(id="title-box"):
                yield Static("🚀 SYSTEM BENCHMARK", id="title")
                yield Static("CPU & GPU Performance Testing Tool", id="subtitle")
            
            # Status Panel
            with Container(id="status-panel"):
                with Horizontal(classes="status-row"):
                    yield Static("Virtual Environment:", classes="status-label")
                    yield Static(self.venv_status, id="venv-status", classes="status-value")
                
                with Horizontal(classes="status-row"):
                    yield Static("GPU Status:", classes="status-label")
                    yield Static(self.gpu_status, id="gpu-status", classes="status-value")
                
                with Horizontal(classes="status-row"):
                    yield Static("Last CPU Score:", classes="status-label")
                    yield Static("Not run", id="cpu-score", classes="status-value")
                
                with Horizontal(classes="status-row"):
                    yield Static("Last GPU Score:", classes="status-label")
                    yield Static("Not run", id="gpu-score", classes="status-value")
            
            # Buttons
            with Horizontal(id="button-panel"):
                yield Button("🔧 Check/Setup Venv", id="venv-button", classes="action-button")
                yield Button("🖥️ CPU Benchmark", id="cpu-button", classes="action-button")
                yield Button("🎮 GPU Benchmark", id="gpu-button", classes="action-button")
                yield Button("📊 View Results", id="results-button", classes="action-button")
            
            # Log Container
            with Container(id="log-container"):
                yield RichLog(id="benchmark-log", highlight=True, markup=True)
            
            # Progress
            with Container(id="progress-container"):
                yield Static("Ready", id="progress-label")
                yield ProgressBar(id="progress-bar", show_eta=False)
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Called when app is mounted"""
        self.query_one("#progress-bar", ProgressBar).update(total=100, progress=0)
        self.check_venv_status()
        self.log_message("Welcome to System Benchmark!")
        self.log_message("Press 'v' to check/setup virtual environment")
        self.log_message("Press 'c' for CPU benchmark, 'g' for GPU benchmark")
        self.log_message("-" * 50)
    
    def log_message(self, message: str) -> None:
        """Add a message to the log"""
        log_widget = self.query_one("#benchmark-log", RichLog)
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_widget.write(f"[dim]{timestamp}[/dim] {message}")
    
    def update_progress(self, progress: float, label: str = "") -> None:
        """Update progress bar"""
        progress_bar = self.query_one("#progress-bar", ProgressBar)
        progress_label = self.query_one("#progress-label", Static)
        progress_bar.update(progress=progress * 100)
        if label:
            progress_label.update(label)
    
    def check_venv_status(self) -> None:
        """Check if venv exists and update status"""
        venv_status_widget = self.query_one("#venv-status", Static)
        
        if self.venv_manager.venv_exists():
            # Also verify packages are installed
            if self.venv_manager.verify_installation():
                self.venv_status = "✅ Ready"
                self.venv_ready = True
                venv_status_widget.update("✅ Ready")
                venv_status_widget.add_class("status-ok")
                venv_status_widget.remove_class("status-error")
                venv_status_widget.remove_class("status-warn")
                
                # Check GPU
                self.check_gpu_status()
            else:
                self.venv_status = "⚠️ Incomplete"
                self.venv_ready = False
                venv_status_widget.update("⚠️ Packages missing - Press 'v' to fix")
                venv_status_widget.add_class("status-warn")
                venv_status_widget.remove_class("status-ok")
                venv_status_widget.remove_class("status-error")
        else:
            self.venv_status = "❌ Not found"
            self.venv_ready = False
            venv_status_widget.update("❌ Not found - Press 'v' to setup")
            venv_status_widget.add_class("status-error")
            venv_status_widget.remove_class("status-ok")
            venv_status_widget.remove_class("status-warn")
    
    @work(exclusive=True, thread=True)
    def check_gpu_status(self) -> None:
        """Check GPU availability in background"""
        gpu_status_widget = self.query_one("#gpu-status", Static)
        
        self.call_from_thread(gpu_status_widget.update, "Checking...")
        
        gpu_info = self.venv_manager.check_gpu_availability()
        
        backend = gpu_info.get("backend", "error")
        
        if backend == "cuda":
            gpu_names = gpu_info.get("gpu_names", ["Unknown"])
            gpu_text = f"✅ {gpu_names[0]} (CUDA {gpu_info.get('cuda_version', 'N/A')})"
            self.call_from_thread(gpu_status_widget.update, gpu_text)
            self.call_from_thread(gpu_status_widget.add_class, "status-ok")
        elif backend == "mps":
            gpu_names = gpu_info.get("gpu_names", ["Apple Silicon"])
            memory = gpu_info.get("gpu_memory", [0])[0]
            gpu_text = f"✅ {gpu_names[0]} ({memory:.0f}GB RAM)"
            self.call_from_thread(gpu_status_widget.update, gpu_text)
            self.call_from_thread(gpu_status_widget.add_class, "status-ok")
        else:
            error = gpu_info.get("error", "No GPU acceleration found")
            self.call_from_thread(gpu_status_widget.update, f"⚠️ CPU only - {error}")
            self.call_from_thread(gpu_status_widget.add_class, "status-warn")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        button_id = event.button.id
        
        if button_id == "venv-button":
            self.action_check_venv()
        elif button_id == "cpu-button":
            self.action_run_cpu()
        elif button_id == "gpu-button":
            self.action_run_gpu()
        elif button_id == "results-button":
            self.action_show_results()
    
    def action_check_venv(self) -> None:
        """Check or setup virtual environment"""
        if self.benchmark_running:
            self.log_message("⚠️ A benchmark is currently running")
            return
        
        if self.venv_manager.venv_exists() and self.venv_manager.verify_installation():
            self.log_message("✅ Virtual environment 'benchmark' exists and is ready")
            self.check_venv_status()
            
            # Offer to recreate
            def handle_recreate(recreate: bool) -> None:
                if recreate:
                    self.log_message("Recreating virtual environment...")
                    self.venv_manager.delete_venv()
                    self.push_screen(SetupScreen(self.venv_manager), self.on_setup_complete)
            
            self.push_screen(
                ConfirmScreen(
                    "⚠️ Recreate Environment?",
                    "Virtual environment exists. Do you want to recreate it?\nThis will delete and reinstall all packages."
                ),
                handle_recreate
            )
        else:
            if self.venv_manager.venv_exists():
                self.log_message("Virtual environment exists but is incomplete. Fixing...")
                self.venv_manager.delete_venv()
            else:
                self.log_message("Virtual environment not found. Starting setup...")
            self.push_screen(SetupScreen(self.venv_manager), self.on_setup_complete)
    
    def on_setup_complete(self, success: bool) -> None:
        """Called when setup screen is dismissed"""
        self.check_venv_status()
        if success:
            self.log_message("✅ Virtual environment setup complete")
        else:
            self.log_message("Virtual environment setup cancelled or failed")
    
    def action_run_cpu(self) -> None:
        """Run CPU benchmark"""
        if self.benchmark_running:
            self.log_message("⚠️ A benchmark is already running")
            return
        
        self.run_cpu_benchmark()
    
    @work(exclusive=True, thread=True)
    def run_cpu_benchmark(self) -> None:
        """Run CPU benchmark in worker thread"""
        self.benchmark_running = True
        
        def log_callback(message: str):
            self.call_from_thread(self.log_message, message)
        
        def progress_callback(progress: float):
            self.call_from_thread(self.update_progress, progress, f"CPU Benchmark: {progress*100:.0f}%")
        
        try:
            benchmark = MandelbrotBenchmark()
            benchmark.set_callbacks(log_callback, progress_callback)
            
            self.cpu_result = benchmark.run_benchmark()
            
            # Update score display
            cpu_score_widget = self.query_one("#cpu-score", Static)
            self.call_from_thread(
                cpu_score_widget.update, 
                f"⭐ {self.cpu_result.score:,.0f}"
            )
            
            self.call_from_thread(self.update_progress, 0, "CPU Benchmark Complete!")
            
        except Exception as e:
            self.call_from_thread(self.log_message, f"❌ CPU Benchmark failed: {e}")
        
        finally:
            self.benchmark_running = False
    
    def action_run_gpu(self) -> None:
        """Run GPU benchmark"""
        if self.benchmark_running:
            self.log_message("⚠️ A benchmark is already running")
            return
        
        if not self.venv_manager.venv_exists():
            self.log_message("❌ Virtual environment not found. Press 'v' to set it up first.")
            return
        
        if not self.venv_manager.verify_installation():
            self.log_message("❌ Virtual environment is incomplete. Press 'v' to fix it.")
            return
        
        self.run_gpu_benchmark()
    
    @work(exclusive=True, thread=True)
    def run_gpu_benchmark(self) -> None:
        """Run GPU benchmark in worker thread"""
        self.benchmark_running = True
        
        def log_callback(message: str):
            self.call_from_thread(self.log_message, message)
        
        def progress_callback(progress: float):
            self.call_from_thread(self.update_progress, progress, f"GPU Benchmark: {progress*100:.0f}%")
        
        try:
            benchmark = LLMBenchmark(self.venv_manager.venv_path)
            benchmark.set_callbacks(log_callback, progress_callback)
            
            self.gpu_result = benchmark.run_benchmark()
            
            # Update score display
            gpu_score_widget = self.query_one("#gpu-score", Static)
            if self.gpu_result.success:
                self.call_from_thread(
                    gpu_score_widget.update, 
                    f"⭐ {self.gpu_result.tokens_per_second:.2f} tok/s"
                )
            else:
                self.call_from_thread(gpu_score_widget.update, "❌ Failed")
            
            self.call_from_thread(self.update_progress, 0, "GPU Benchmark Complete!")
            
        except Exception as e:
            self.call_from_thread(self.log_message, f"❌ GPU Benchmark failed: {e}")
        
        finally:
            self.benchmark_running = False
    
    def action_show_results(self) -> None:
        """Show results screen"""
        self.push_screen(ResultsScreen(self.cpu_result, self.gpu_result))
    
    def action_toggle_dark(self) -> None:
        """Toggle dark mode"""
        self.dark = not self.dark


def main():
    """Main entry point"""
    app = BenchmarkApp()
    app.run()


if __name__ == "__main__":
    main()