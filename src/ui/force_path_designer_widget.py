"""Force Path Designer Widget for designing frequency and amplitude sequences over time."""

from typing import Optional, List, Dict, Any
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QLabel, QGroupBox, QHeaderView, QComboBox, QDoubleSpinBox,
    QSpinBox, QMessageBox, QFrame, QSplitter, QTextEdit,
    QMainWindow
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QFont
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import time

from src.utils.logger import get_logger
from src.utils.status_display import StatusDisplay

logger = get_logger("force_path_designer")


class TransitionType(Enum):
    """Types of transitions between path points."""
    HOLD = "Hold"
    LINEAR = "Linear"


@dataclass
class PathPoint:
    """Single point in the force path."""
    time: float  # Time in seconds
    frequency: float  # Frequency in MHz
    amplitude: float  # Amplitude in Vpp
    transition: TransitionType = TransitionType.LINEAR


class FunctionGeneratorWorker(QThread):
    """Worker thread for non-blocking function generator communication."""
    
    # Signals for status updates
    status_update = pyqtSignal(str, float, float)  # status, frequency, amplitude
    execution_finished = pyqtSignal(bool)  # success
    
    def __init__(self, fg_controller, sorted_points, execution_log, main_window=None):
        super().__init__()
        self.fg_controller = fg_controller
        self.sorted_points = sorted_points
        self.execution_log = execution_log
        self.main_window = main_window  # Reference for HDF5 timeline logging
        self.should_stop = False
        self.measurement_start_time = time.time()
        
    def stop(self):
        """Request worker to stop."""
        self.should_stop = True
        
    def run(self):
        """Execute the path in a separate thread."""
        try:
            if not self.fg_controller or not self.fg_controller.is_connected:
                self.execution_finished.emit(False)
                return
                
            start_time = time.perf_counter()
            total_duration = self.sorted_points[-1].time if self.sorted_points else 0
            
            # Debug logging of received points
            logger.info(f"FunctionGeneratorWorker received {len(self.sorted_points)} points:")
            for i, point in enumerate(self.sorted_points):
                logger.info(f"  Worker Point {i}: time={point.time:.1f}s, freq={point.frequency:.3f}MHz, amp={point.amplitude:.2f}Vpp")
            
            # Process points to handle time 0 hold behavior
            processed_points = self._process_points_for_execution()
            
            logger.info(f"FunctionGeneratorWorker processed to {len(processed_points)} points:")
            for i, point in enumerate(processed_points):
                logger.info(f"  Processed Point {i}: time={point.time:.1f}s, freq={point.frequency:.3f}MHz, amp={point.amplitude:.2f}Vpp")
            
            logger.info(f"Total execution duration: {total_duration:.1f}s")
            
            # Start output with first point values (turn ON once)
            if processed_points:
                first_point = processed_points[0]
                success = self.fg_controller.output_sine_wave(
                    amplitude=first_point.amplitude,
                    frequency_mhz=first_point.frequency,
                    channel=1
                )
                if not success:
                    logger.error("Failed to start function generator output")
                    self.execution_finished.emit(False)
                    return
            
            # Main execution loop - only update parameters, keep output ON
            while not self.should_stop:
                elapsed_time = time.perf_counter() - start_time
                
                # Check if execution is complete
                if elapsed_time >= total_duration:
                    break
                    
                # Find current interpolation segment
                current_idx = 0
                for i, point in enumerate(processed_points):
                    if point.time <= elapsed_time:
                        current_idx = i
                    else:
                        break
                
                # Calculate interpolated values
                if current_idx < len(processed_points) - 1:
                    point1 = processed_points[current_idx]
                    point2 = processed_points[current_idx + 1]
                    
                    if point1.time == point2.time:
                        progress = 0.0
                    else:
                        progress = (elapsed_time - point1.time) / (point2.time - point1.time)
                        progress = max(0.0, min(1.0, progress))
                    
                    if point2.transition == TransitionType.HOLD:
                        frequency = point1.frequency
                        amplitude = point1.amplitude
                    else:  # LINEAR
                        frequency = point1.frequency + progress * (point2.frequency - point1.frequency)
                        amplitude = point1.amplitude + progress * (point2.amplitude - point1.amplitude)
                    
                    # Update function generator parameters only (keep output ON)
                    # Use delta-based updates to reduce unnecessary commands
                    needs_update = False
                    if not hasattr(self, '_last_freq') or abs(self._last_freq - frequency) > 0.005:
                        needs_update = True
                    if not hasattr(self, '_last_amp') or abs(self._last_amp - amplitude) > 0.005:
                        needs_update = True
                    
                    if needs_update:
                        # Try batch update first (fastest), fallback to regular if needed
                        success = self.fg_controller.update_parameters_batch(
                            amplitude=amplitude,
                            frequency_mhz=frequency,
                            channel=1
                        )
                        
                        if success:
                            self._last_freq = frequency
                            self._last_amp = amplitude
                            
                            # Reset consecutive error counter
                            if hasattr(self, '_consecutive_errors'):
                                self._consecutive_errors = 0
                            
                            # ENHANCED LOGGING - Detailed function generator output tracking
                            import datetime
                            current_timestamp = time.time()
                            log_entry = {
                                'execution_time_s': elapsed_time,
                                'absolute_timestamp': current_timestamp,
                                'iso_timestamp': datetime.datetime.fromtimestamp(current_timestamp).isoformat(),
                                'set_frequency_mhz': frequency,
                                'set_amplitude_vpp': amplitude,
                                'interpolation_progress': progress,
                                'current_segment': f"{current_idx}->{current_idx+1}",
                                'segment_start_time': point1.time,
                                'segment_end_time': point2.time,
                                'transition_type': point2.transition.value,
                                'delta_freq_mhz': abs(self._last_freq - frequency) if hasattr(self, '_last_freq') else 0.0,
                                'delta_amp_vpp': abs(self._last_amp - amplitude) if hasattr(self, '_last_amp') else 0.0
                            }
                            self.execution_log.append(log_entry)
                            
                            # Real-time console logging for immediate feedback
                            logger.info(f"FG Output: {elapsed_time:.2f}s -> {frequency:.3f}MHz, {amplitude:.2f}Vpp")
                            
                            # MAIN WINDOW HDF5 TIMELINE LOGGING - Log to main recording system
                            if hasattr(self, 'main_window') and self.main_window and hasattr(self.main_window, 'camera_widget'):
                                try:
                                    camera_widget = self.main_window.camera_widget
                                    if hasattr(camera_widget, 'log_function_generator_event'):
                                        # Log function generator output to main HDF5 timeline
                                        camera_widget.log_function_generator_event(
                                            frequency_mhz=frequency,
                                            amplitude_vpp=amplitude,
                                            output_enabled=True,  # Always true during force path execution
                                            event_type='force_path_execution'
                                        )
                                except Exception as e:
                                    # Don't let logging errors interrupt execution
                                    logger.debug(f"Failed to log FG event to main HDF5: {e}")
                            
                            # Emit status update (reduced frequency to minimize UI overhead)
                            if not hasattr(self, '_last_status_time') or (elapsed_time - self._last_status_time) > 0.2:
                                # Simplified status message - detailed info in logs
                                self.status_update.emit(f"Executing", frequency, amplitude)
                                self._last_status_time = elapsed_time
                        else:
                            # Handle communication errors with retry logic
                            if not hasattr(self, '_consecutive_errors'):
                                self._consecutive_errors = 0
                            self._consecutive_errors += 1
                            
                            if self._consecutive_errors > 3:  # Allow a few retries
                                logger.error("Function generator communication failed during execution")
                                self.execution_finished.emit(False)
                                return
                            else:
                                logger.warning(f"Function generator retry {self._consecutive_errors}/3")
                                # Brief pause before retry
                                self.msleep(10)
                
                # Optimized sleep - balanced speed vs USB communication limits
                self.msleep(20)  # 20ms = 50 Hz update rate (optimal for USB VISA)
            
            # CRITICAL: Turn off function generator output after execution completes
            logger.info("Force path execution completed - turning off function generator output")
            try:
                self.fg_controller.stop_all_outputs()
                logger.info("Function generator output turned off successfully")
            except Exception as e:
                logger.error(f"Failed to turn off function generator output after execution: {e}")
                
            self.execution_finished.emit(True)
            
        except Exception as e:
            logger.error(f"Function generator worker error: {e}")
            # Try to turn off output even on error
            try:
                if self.fg_controller and self.fg_controller.is_connected:
                    self.fg_controller.stop_all_outputs()
                    logger.info("Function generator output turned off after error")
            except:
                pass
            self.execution_finished.emit(False)
            
    def _process_points_for_execution(self):
        """Process points to handle time 0 hold behavior."""
        if not self.sorted_points:
            return []
            
        processed_points = self.sorted_points.copy()
        
        # Check if first point is not at time 0
        first_point = processed_points[0]
        if first_point.time > 0:
            # Add a hold point from 0 to first point time
            hold_point = PathPoint(
                time=0.0,
                frequency=first_point.frequency,
                amplitude=first_point.amplitude,
                transition=TransitionType.HOLD
            )
            processed_points.insert(0, hold_point)
            
        return processed_points


class ForcePathDesignerWidget(QWidget):
    """Widget for designing force paths with frequency and amplitude control over time."""
    
    # Signals
    path_loaded = pyqtSignal(list)  # Emitted when a path is loaded
    path_execution_started = pyqtSignal()
    path_execution_stopped = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Path data
        self.path_points: List[PathPoint] = []
        
        # Execution state
        self.is_executing = False
        self.execution_timer: Optional[QTimer] = None
        self.execution_start_time = 0.0
        self.current_point_index = 0
        self.current_execution_time = 0.0  # For live line tracking
        
        # Function generator reference (set from parent)
        self.function_generator_controller = None
        
        # HDF5 export data - automatic logging
        self.execution_log: List[Dict] = []  # Log of actual vs set values during execution
        self.main_window = None  # Reference to main window for measurement logging
        self.measurement_start_time: Optional[float] = None  # When measurement was started
        self.execution_completed: bool = False  # Whether execution finished normally
        
        # Performance optimizations
        self._last_graph_hash = None  # Cache for graph updates
        self._graph_update_cache = {}  # Cache computed values
        
        # Live execution tracking
        self.live_line = None  # Vertical line showing current execution position
        
        self._init_ui()
        self._add_default_points()
        
        # Initialize the graph
        self._update_graph()
        
    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(5)  # Minimal spacing
        layout.setContentsMargins(5, 5, 5, 5)  # Minimal margins
        
        # Create main splitter
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        # Left side: Table and controls
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(5)
        
        # Path table section
        table_widget = self._create_table_section()
        left_layout.addWidget(table_widget)
        
        # Control buttons section
        controls_widget = self._create_controls_section()
        left_layout.addWidget(controls_widget)
        
        splitter.addWidget(left_widget)
        
        # Right side: Graph
        graph_widget = self._create_graph_section()
        splitter.addWidget(graph_widget)
        
        # Set splitter proportions - table on left, graph on right (wider layout)
        splitter.setSizes([400, 600])
        
        # Set minimal styling - remove all borders
        self.setStyleSheet("""
            QWidget {
                border: none;
                margin: 2px;
                padding: 2px;
            }
            QPushButton {
                padding: 4px 8px;
                border: 1px solid #c0c0c0;
                border-radius: 2px;
                background-color: #f8f8f8;
                min-height: 18px;
            }
            QPushButton:hover {
                background-color: #e8e8e8;
                border-color: #a0a0a0;
            }
            QPushButton:pressed {
                background-color: #d8d8d8;
            }
            QPushButton:disabled {
                background-color: #f5f5f5;
                color: #a0a0a0;
                border-color: #e0e0e0;
            }
            QTableWidget {
                border: 1px solid #d0d0d0;
                gridline-color: #e0e0e0;
                background-color: white;
            }
            QTableWidget::item {
                padding: 4px;
                border: none;
            }
            QTableWidget::item:selected {
                background-color: #e0f0ff;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                padding: 4px 8px;
                border: none;
                border-right: 1px solid #d0d0d0;
                border-bottom: 1px solid #d0d0d0;
                font-weight: normal;
            }
        """)
        
    def _create_table_section(self) -> QWidget:
        """Create the path table section."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        
        # Create table with clear headers - use seconds for consistency
        self.path_table = QTableWidget()
        self.path_table.setColumnCount(3)
        self.path_table.setHorizontalHeaderLabels([
            "Duration (s)", "Frequency (MHz)", "Amplitude (Vpp)"
        ])
        
        # Add tooltip for time column to explain relative behavior
        self.path_table.horizontalHeaderItem(0).setToolTip(
            "Duration in seconds from previous point\n"
            "First point: time from start (0)\n"
            "Subsequent points: time since previous point\n"
            "Example: [5, 6, 3] → total execution time = 14 seconds"
        )
        
        # Configure table for minimal appearance
        header = self.path_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        
        self.path_table.setAlternatingRowColors(True)
        self.path_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.path_table.setShowGrid(True)
        
        layout.addWidget(self.path_table)
        
        # Table control buttons - simplified layout
        table_controls = QHBoxLayout()
        table_controls.setSpacing(8)
        table_controls.setContentsMargins(0, 5, 0, 5)
        
        self.add_point_btn = QPushButton("Add")
        self.add_point_btn.setMaximumWidth(60)
        table_controls.addWidget(self.add_point_btn)
        
        self.remove_point_btn = QPushButton("Remove")
        self.remove_point_btn.setMaximumWidth(70)
        table_controls.addWidget(self.remove_point_btn)
        
        self.clear_path_btn = QPushButton("Clear")
        self.clear_path_btn.setMaximumWidth(60)
        table_controls.addWidget(self.clear_path_btn)
        
        table_controls.addStretch()
        
        layout.addLayout(table_controls)
        
        # Connect signals
        self.add_point_btn.clicked.connect(self._add_point)
        self.remove_point_btn.clicked.connect(self._remove_point)
        self.clear_path_btn.clicked.connect(self._clear_path)
        
        return widget
        
    def _determine_transition(self, point_index: int) -> TransitionType:
        """Automatically determine transition type based on value changes (optimized)."""
        if point_index == 0 or point_index >= len(self.path_points):
            return TransitionType.HOLD
            
        current = self.path_points[point_index]
        previous = self.path_points[point_index - 1]
        
        # Use more efficient comparison (avoid abs() for performance)
        freq_diff = current.frequency - previous.frequency
        amp_diff = current.amplitude - previous.amplitude
        
        # Check if both values are essentially the same (tolerance: 0.001)
        if freq_diff * freq_diff < 1e-6 and amp_diff * amp_diff < 1e-6:
            return TransitionType.HOLD
        else:
            return TransitionType.LINEAR
            
    def _update_transitions(self):
        """Update all transition types automatically (vectorized for performance)."""
        # Process transitions in batch for better performance
        for i in range(len(self.path_points)):
            self.path_points[i].transition = self._determine_transition(i)
        
    def _create_graph_section(self) -> QWidget:
        """Create the minimal graph section with optimized performance."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Create matplotlib figure with performance optimizations
        self.figure = Figure(figsize=(6, 4), dpi=80, facecolor='white')
        self.figure.patch.set_facecolor('white')
        
        # Use optimized canvas for better performance
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setParent(widget)
        
        # Create single subplot with dual y-axes
        self.ax1 = self.figure.add_subplot(111)
        self.ax2 = self.ax1.twinx()  # Create second y-axis sharing same x-axis
        
        # Performance optimizations
        self.ax1.set_rasterization_zorder(0)  # Rasterize for speed
        self.ax2.set_rasterization_zorder(0)
        
        # Set up empty plot initially
        self._setup_empty_plot()
        
        layout.addWidget(self.canvas)
        
        return widget
        
    def _setup_empty_plot(self):
        """Set up empty plot with proper styling and default ranges."""
        self.ax1.clear()
        self.ax2.clear()
        
        # Set axis labels - consistent with seconds
        self.ax1.set_xlabel('Time (s)', fontsize=10)
        self.ax1.set_ylabel('Amplitude (Vpp)', color='red', fontsize=10)
        self.ax2.set_ylabel('Frequency (MHz)', color='blue', fontsize=10)
        
        # Move frequency label to the right
        self.ax2.yaxis.set_label_position('right')
        self.ax2.yaxis.tick_right()
        
        # Color the y-axis ticks to match the lines
        self.ax1.tick_params(axis='y', labelcolor='red', labelsize=8)
        self.ax2.tick_params(axis='y', labelcolor='blue', labelsize=8)
        self.ax1.tick_params(axis='x', labelsize=8)
        
        # Remove top and right spines for cleaner look
        self.ax1.spines['top'].set_visible(False)
        self.ax2.spines['top'].set_visible(False)
        
        # Set default ranges: 13-15 MHz frequency, 1-4 Vpp amplitude, 0-300 seconds time
        self.ax1.set_xlim(left=0, right=300)  # Default time range: 0 to 5 minutes (300s)
        self.ax1.set_ylim(bottom=1.0, top=4.0)  # Default amplitude range 1-4 Vpp
        self.ax2.set_ylim(bottom=13.0, top=15.0)  # Default frequency range 13-15 MHz
        
        # Remove grid
        self.ax1.grid(False)
        self.ax2.grid(False)
        
        self.figure.tight_layout(pad=1.0)
        self.canvas.draw()
        
    def _create_controls_section(self) -> QWidget:
        """Create the control buttons section."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 5, 0, 5)
        layout.setSpacing(10)
        
        # Execution control buttons - more compact
        self.execute_path_btn = QPushButton("Execute Path")
        self.execute_path_btn.setMinimumWidth(100)
        self.execute_path_btn.clicked.connect(self._execute_path)
        layout.addWidget(self.execute_path_btn)
        
        self.stop_execution_btn = QPushButton("Stop")
        self.stop_execution_btn.setMinimumWidth(60)
        self.stop_execution_btn.clicked.connect(self._stop_execution)
        self.stop_execution_btn.setEnabled(False)
        layout.addWidget(self.stop_execution_btn)
        
        layout.addStretch()  # Push buttons to the left
        
        # Status display using standardized StatusDisplay widget
        self.status_display = StatusDisplay()
        self.status_display.set_status("Disconnected")  # Default to disconnected until FG controller is set
        layout.addWidget(self.status_display)
        
        return widget
        
    def _add_default_points(self):
        """Add default starting point at time 0."""
        # Start with a default point at time 0
        default_point = PathPoint(
            time=0.0,
            frequency=14.0,  # Default 14 MHz
            amplitude=4.0,   # Default 4 Vpp
            transition=TransitionType.LINEAR
        )
        self.path_points = [default_point]
        self._update_table()
        self._update_graph()
        
    def _add_point(self):
        """Add a new point to the path with 30-second increments."""
        # Get time for new point (after last point)
        if not self.path_points:
            # Should not happen with default point, but just in case
            last_time = 0.0
        else:
            last_time = max(point.time for point in self.path_points)
        
        new_point = PathPoint(
            time=last_time + 30.0,  # Increment by 30s from last point
            frequency=14.0,  # Default 14 MHz
            amplitude=4.0,   # Default 4 Vpp
            transition=TransitionType.LINEAR
        )
        
        self.path_points.append(new_point)
        self._update_table()
        self._update_graph()
        
    def _remove_point(self):
        """Remove selected point from the path."""
        current_row = self.path_table.currentRow()
        if current_row >= 0 and current_row < len(self.path_points):
            self.path_points.pop(current_row)
            self._update_table()
            self._update_graph()
            
    def _clear_path(self):
        """Clear all points from the path."""
        reply = QMessageBox.question(
            self, "Clear Path", 
            "Are you sure you want to clear all path points?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.path_points.clear()
            self._update_table()
            self._update_graph()
            
    def _update_table(self):
        """Update the table with current path points (optimized for performance)."""
        # Temporarily disconnect signals to avoid recursion during bulk updates
        try:
            self.path_table.itemChanged.disconnect()
        except TypeError:
            pass  # No connections to disconnect
        try:
            self.path_table.cellChanged.disconnect()
        except TypeError:
            pass  # No connections to disconnect
        
        # Update transitions automatically
        self._update_transitions()
        
        self.path_table.setRowCount(len(self.path_points))
        
        # Batch update for better performance
        for row, point in enumerate(self.path_points):
            # Time - show relative duration (time from previous point)
            relative_time = self._get_relative_time(row)
            time_item = QTableWidgetItem(f"{relative_time:.1f}")
            time_item.setData(Qt.UserRole, relative_time)  # Store relative value for editing
            self.path_table.setItem(row, 0, time_item)
            
            # Frequency - 3 decimal places for precision
            freq_item = QTableWidgetItem(f"{point.frequency:.3f}")
            self.path_table.setItem(row, 1, freq_item)
            
            # Amplitude - 2 decimal places
            amp_item = QTableWidgetItem(f"{point.amplitude:.2f}")
            self.path_table.setItem(row, 2, amp_item)
            
        # Reconnect signals
        self.path_table.itemChanged.connect(self._on_table_item_changed)
        self.path_table.cellChanged.connect(self._on_cell_changed)
    
    def _get_relative_time(self, point_index: int) -> float:
        """Get relative time for a point (duration from previous point)."""
        if point_index == 0:
            # First point is always at absolute time, show as relative from 0
            return self.path_points[0].time
        else:
            # Return difference from previous point
            current_time = self.path_points[point_index].time
            previous_time = self.path_points[point_index - 1].time
            return current_time - previous_time
    
    def _set_relative_time(self, point_index: int, relative_time: float):
        """Set time for a point based on relative duration."""
        if point_index == 0:
            # First point: relative time is the absolute time
            self.path_points[0].time = relative_time
        else:
            # Subsequent points: add relative time to previous absolute time
            previous_absolute = self.path_points[point_index - 1].time
            self.path_points[point_index].time = previous_absolute + relative_time
        
        # Update all subsequent points to maintain their relative durations
        for i in range(point_index + 1, len(self.path_points)):
            if i > 0:
                # Keep the same relative duration for this point
                relative_duration = self._get_relative_time(i)
                self.path_points[i].time = self.path_points[i - 1].time + relative_duration
        
    def _parse_time_value(self, text: str) -> float:
        """Parse time value from text in seconds."""
        if not text:
            return 0.0
            
        text = text.strip()
        
        # Handle MM:SS format for backward compatibility
        if ':' in text:
            parts = text.split(':')
            if len(parts) == 2:  # MM:SS
                minutes, seconds = map(int, parts)
                return minutes * 60 + seconds
            elif len(parts) == 3:  # HH:MM:SS
                hours, minutes, seconds = map(int, parts)
                return hours * 3600 + minutes * 60 + seconds
            else:
                raise ValueError("Invalid time format")
        else:
            # Handle numeric format in seconds
            return float(text)

    def _on_table_item_changed(self, item):
        """Handle table item changes with relative time support."""
        row = item.row()
        col = item.column()
        
        if row >= len(self.path_points):
            return
            
        try:
            if col == 0:  # Time (relative duration)
                # Always parse from text for user edits
                relative_time = self._parse_time_value(item.text())
                
                # Debug logging
                old_relative = self._get_relative_time(row)
                old_absolute = self.path_points[row].time
                
                # Set the relative time, which updates absolute times
                self._set_relative_time(row, relative_time)
                
                new_absolute = self.path_points[row].time
                logger.debug(f"Time updated for point {row}: relative {old_relative:.1f}s -> {relative_time:.1f}s (absolute {old_absolute:.1f}s -> {new_absolute:.1f}s)")
                
                # Update stored data to match the relative value
                item.setData(Qt.UserRole, relative_time)
                
                # Need to refresh table to show updated relative times for subsequent points
                self._update_table()
                return  # Skip normal update since we called _update_table()
                
            elif col == 1:  # Frequency
                old_freq = self.path_points[row].frequency
                new_freq = float(item.text())
                self.path_points[row].frequency = new_freq
                logger.debug(f"Frequency updated for point {row}: {old_freq:.3f}MHz -> {new_freq:.3f}MHz")
                
            elif col == 2:  # Amplitude
                old_amp = self.path_points[row].amplitude
                new_amp = float(item.text())
                self.path_points[row].amplitude = new_amp
                logger.debug(f"Amplitude updated for point {row}: {old_amp:.2f}Vpp -> {new_amp:.2f}Vpp")
                
            # Auto-update transitions for this and subsequent points
            self._update_transitions()
            self._update_graph()
            
        except (ValueError, TypeError) as e:
            # Restore original value if invalid input
            logger.warning(f"Invalid input in table cell ({row}, {col}): {e}")
            self._update_table()
            
    def _on_cell_changed(self, row: int, col: int):
        """Handle individual cell changes for real-time graph updates with relative time support."""
        if row >= len(self.path_points):
            return
        
        # Get the current item value
        item = self.path_table.item(row, col)
        if not item:
            return
            
        try:
            # Update data structure efficiently
            point = self.path_points[row]
            if col == 0:  # Time (relative duration)
                # Always parse from text for user edits
                relative_time = self._parse_time_value(item.text())
                # Set the relative time, which updates absolute times
                self._set_relative_time(row, relative_time)
                # Update stored data to match
                item.setData(Qt.UserRole, relative_time)
            elif col == 1:  # Frequency
                # Parse value with better error handling
                value = float(item.text()) if item.text() else 14.0
                # Clamp frequency to reasonable range (10-20 MHz)
                point.frequency = max(10.0, min(20.0, value))
            elif col == 2:  # Amplitude
                # Parse value with better error handling
                value = float(item.text()) if item.text() else 4.0
                # Clamp amplitude to reasonable range (0.5-5.0 V)
                point.amplitude = max(0.5, min(5.0, value))
                
            # Batch updates for performance
            self._update_transitions()
            
            # Debounced graph update (use timer for rapid changes)
            if not hasattr(self, '_update_timer'):
                from PyQt5.QtCore import QTimer
                self._update_timer = QTimer()
                self._update_timer.setSingleShot(True)
                self._update_timer.timeout.connect(self._update_graph)
            
            self._update_timer.stop()
            self._update_timer.start(50)  # 50ms debounce for smooth performance
            
        except (ValueError, TypeError):
            # Invalid input, ignore for now (will be restored by table update)
            pass
            
    def _compute_path_hash(self) -> str:
        """Compute a hash of the current path for caching purposes."""
        import hashlib
        path_data = []
        for point in self.path_points:
            path_data.extend([point.time, point.frequency, point.amplitude])
        
        # Create hash from the data
        data_str = ','.join(f"{x:.6f}" for x in path_data)
        return hashlib.md5(data_str.encode()).hexdigest()
        
    def _update_graph(self):
        """Update the path graph with optimized performance and caching."""
        # Check if graph needs updating using hash comparison
        current_hash = self._compute_path_hash()
        if current_hash == self._last_graph_hash and current_hash in self._graph_update_cache:
            # Use cached data for extremely fast updates
            cached_data = self._graph_update_cache[current_hash]
            self._render_graph_from_cache(cached_data)
            return
        
        if not self.path_points:
            # Show empty plot with default ranges
            self._setup_empty_plot()
            self._last_graph_hash = current_hash
            return
            
        # Sort points by time (optimized)
        sorted_points = sorted(self.path_points, key=lambda p: p.time)
        
        # Add hold transition from 0 if first point doesn't start at 0
        if sorted_points and sorted_points[0].time > 0:
            # Add origin point with same values as first point but at time 0 (hold transition)
            first_point = sorted_points[0]
            origin_point = PathPoint(0.0, first_point.frequency, first_point.amplitude, TransitionType.HOLD)
            sorted_points.insert(0, origin_point)
        
        # Extract data for plotting (vectorized for performance)
        times = [point.time for point in sorted_points]
        frequencies = [point.frequency for point in sorted_points]
        amplitudes = [point.amplitude for point in sorted_points]
        
        # Calculate ranges for better scaling with default ranges
        freq_min = min(min(frequencies), 13.0)  # Default minimum 13 MHz
        freq_max = max(max(frequencies), 15.0)  # Default maximum 15 MHz
        amp_min = min(min(amplitudes), 1.0)     # Default minimum 1 Vpp
        amp_max = max(max(amplitudes), 4.0)     # Default maximum 4 Vpp
        
        # Add padding to prevent overlap (5% padding for better visibility)
        freq_range = freq_max - freq_min
        amp_range = amp_max - amp_min
        
        freq_padding = max(0.2, freq_range * 0.05)  # Minimum 0.2 MHz padding
        amp_padding = max(0.2, amp_range * 0.05)    # Minimum 0.2 V padding
        
        # Calculate time padding
        time_max = max(times) if times else 300
        if time_max > 300:  # More than 5 minutes
            time_padding = time_max * 0.05  # 5% padding
        else:
            time_padding = max(30, time_max * 0.1)  # At least 30s or 10% padding
        
        # Clear and plot (batch operations for performance)
        self.ax1.clear()
        self.ax2.clear()
        
        # Plot amplitude on left y-axis (red) - highly visible dashed lines
        self.ax1.plot(times, amplitudes, 'r--', linewidth=3, label='Amplitude', 
                     antialiased=True, alpha=1.0, dash_capstyle='round', 
                     dashes=[8, 4])  # Custom dash pattern: 8pt line, 4pt gap
        self.ax1.plot(times, amplitudes, 'ro', markersize=6, markerfacecolor='red', 
                     markeredgecolor='darkred', markeredgewidth=1)
        
        # Plot frequency on right y-axis (blue) - contrasting solid lines
        self.ax2.plot(times, frequencies, 'b-', linewidth=3, label='Frequency', 
                     antialiased=True, alpha=1.0)
        self.ax2.plot(times, frequencies, 'bs', markersize=6, markerfacecolor='blue',
                     markeredgecolor='darkblue', markeredgewidth=1)  # Square markers for distinction
        
        # Set axis labels - use seconds consistently
        self.ax1.set_xlabel('Time (s)', fontsize=10)
        self.ax1.set_ylabel('Amplitude (Vpp)', color='red', fontsize=10)
        self.ax2.set_ylabel('Frequency (MHz)', color='blue', fontsize=10)
        
        # Move frequency label to the right
        self.ax2.yaxis.set_label_position('right')
        self.ax2.yaxis.tick_right()
        
        # Color the y-axis ticks to match the lines
        self.ax1.tick_params(axis='y', labelcolor='red', labelsize=8)
        self.ax2.tick_params(axis='y', labelcolor='blue', labelsize=8)
        self.ax1.tick_params(axis='x', labelsize=8)
        
        # Remove top spines for cleaner look
        self.ax1.spines['top'].set_visible(False)
        self.ax2.spines['top'].set_visible(False)
        
        # Set axis limits with proper scaling to prevent overlap and default ranges
        time_max = max(times) if times else 50
        # For longer durations, add reasonable padding
        if time_max > 300:  # More than 5 minutes
            time_padding = time_max * 0.05  # 5% padding
        else:
            time_padding = max(30, time_max * 0.1)  # At least 30s or 10% padding
            
        self.ax1.set_xlim(left=0, right=time_max + time_padding)
        self.ax1.set_ylim(bottom=amp_min - amp_padding, top=amp_max + amp_padding)
        self.ax2.set_ylim(bottom=freq_min - freq_padding, top=freq_max + freq_padding)
        
        # Remove grid
        self.ax1.grid(False)
        self.ax2.grid(False)
        
        # Cache the computed data for future use
        cache_data = {
            'times': times,
            'frequencies': frequencies,
            'amplitudes': amplitudes,
            'ranges': (freq_min, freq_max, amp_min, amp_max, time_max),
            'paddings': (freq_padding, amp_padding, time_padding)
        }
        self._graph_update_cache[current_hash] = cache_data
        self._last_graph_hash = current_hash
        
        # Render the graph
        self._render_graph_data(cache_data)
        
    def _render_graph_from_cache(self, cache_data: dict):
        """Render graph from cached data for maximum performance."""
        self._render_graph_data(cache_data)
        
    def _render_graph_data(self, data: dict):
        """Render graph data with optimized matplotlib calls."""
        times = data['times']
        frequencies = data['frequencies']
        amplitudes = data['amplitudes']
        freq_min, freq_max, amp_min, amp_max, time_max = data['ranges']
        freq_padding, amp_padding, time_padding = data['paddings']
        
        # Clear and plot (batch operations for performance)
        self.ax1.clear()
        self.ax2.clear()
        
        # Plot amplitude on left y-axis (red) - highly visible dashed lines
        self.ax1.plot(times, amplitudes, 'r--', linewidth=3, label='Amplitude', 
                     antialiased=True, alpha=1.0, dash_capstyle='round', 
                     dashes=[8, 4])  # Custom dash pattern: 8pt line, 4pt gap
        self.ax1.plot(times, amplitudes, 'ro', markersize=6, markerfacecolor='red', 
                     markeredgecolor='darkred', markeredgewidth=1)
        
        # Plot frequency on right y-axis (blue) - contrasting solid lines
        self.ax2.plot(times, frequencies, 'b-', linewidth=3, label='Frequency', 
                     antialiased=True, alpha=1.0)
        self.ax2.plot(times, frequencies, 'bs', markersize=6, markerfacecolor='blue',
                     markeredgecolor='darkblue', markeredgewidth=1)  # Square markers for distinction
        
        # Set axis labels with better time formatting
        self.ax1.set_xlabel('Time (s)', fontsize=10)
        self.ax1.set_ylabel('Amplitude (Vpp)', color='red', fontsize=10)
        self.ax2.set_ylabel('Frequency (MHz)', color='blue', fontsize=10)
        
        # Format x-axis for better time readability with longer durations
        if time_max > 300:  # More than 5 minutes, use MM:SS format
            import matplotlib.ticker as ticker
            def time_formatter(x, pos):
                minutes = int(x // 60)
                seconds = int(x % 60)
                return f"{minutes:02d}:{seconds:02d}"
            self.ax1.xaxis.set_major_formatter(ticker.FuncFormatter(time_formatter))
            self.ax1.set_xlabel('Time (MM:SS)', fontsize=10)
        else:
            self.ax1.set_xlabel('Time (s)', fontsize=10)
        
        # Move frequency label to the right
        self.ax2.yaxis.set_label_position('right')
        self.ax2.yaxis.tick_right()
        
        # Color the y-axis ticks to match the lines
        self.ax1.tick_params(axis='y', labelcolor='red', labelsize=8)
        self.ax2.tick_params(axis='y', labelcolor='blue', labelsize=8)
        self.ax1.tick_params(axis='x', labelsize=8)
        
        # Remove top spines for cleaner look
        self.ax1.spines['top'].set_visible(False)
        self.ax2.spines['top'].set_visible(False)
        
        # Set axis limits with proper scaling to prevent overlap and default ranges
        self.ax1.set_xlim(left=0, right=time_max + time_padding)
        self.ax1.set_ylim(bottom=amp_min - amp_padding, top=amp_max + amp_padding)
        self.ax2.set_ylim(bottom=freq_min - freq_padding, top=freq_max + freq_padding)
        
        # Remove grid
        self.ax1.grid(False)
        self.ax2.grid(False)
        
        # Optimize rendering
        self.figure.tight_layout(pad=0.8)
        
        # Use blit for faster rendering when possible
        try:
            self.canvas.draw_idle()  # Faster than draw() for real-time updates
        except:
            self.canvas.draw()  # Fallback
            
    def _validate_path(self):
        """Validate the current path silently."""
        if not self.path_points:
            self.status_display.set_status("Error")
            logger.warning("Path validation failed: No path points defined")
            return False
            
        # Sort points by time
        sorted_points = sorted(self.path_points, key=lambda p: p.time)
        
        errors = []
        
        # Check for valid values
        for i, point in enumerate(sorted_points):
            if point.time < 0:
                errors.append(f"Point {i+1}: Time cannot be negative")
            if point.frequency <= 0:
                errors.append(f"Point {i+1}: Frequency must be positive")
            if point.amplitude <= 0:
                errors.append(f"Point {i+1}: Amplitude must be positive")
                
        # Check time ordering
        for i in range(1, len(sorted_points)):
            if sorted_points[i].time <= sorted_points[i-1].time:
                errors.append(f"Point {i+1}: Time must be greater than previous point")
                
        if errors:
            self.status_display.set_status("Error")
            logger.warning(f"Path validation failed: {'; '.join(errors)}")
            return False
        else:
            logger.info(f"Path validation successful: {len(sorted_points)} points, "
                       f"duration: {sorted_points[-1].time:.1f} seconds")
            return True
            
    def _execute_path(self):
        """Execute the current force path using non-blocking approach."""
        if not self._validate_path():
            return
            
        if not self.function_generator_controller:
            self.status_display.set_status("Disconnected")
            logger.error("Execution error: No function generator controller available")
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Function Generator Disconnected", 
                              "Cannot execute force path: Function generator is not connected.\n"
                              "Please check the function generator connection and try again.")
            return
        
        # Try to ensure connection before execution
        if not self.function_generator_controller.is_connected:
            logger.info("Function generator not connected, attempting to connect...")
            # Try to reconnect
            try:
                if not self.function_generator_controller.connect():
                    self.status_display.set_status("Disconnected")
                    logger.error("Function generator connection failed - please check device")
                    from PyQt5.QtWidgets import QMessageBox
                    QMessageBox.warning(self, "Function Generator Connection Failed", 
                                      "Cannot execute force path: Function generator connection failed.\n"
                                      "Please check the device connection and try again.")
                    return
                else:
                    logger.info("Function generator reconnected successfully")
                    self.status_display.set_status("Ready")
            except Exception as e:
                self.status_display.set_status("Disconnected")
                logger.error(f"Function generator connection error: {e}")
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Function Generator Error", 
                                  f"Cannot execute force path: {str(e)}\n"
                                  "Please check the device connection and try again.")
                return
            
        # Prepare execution
        self.is_executing = True
        self.execution_start_time = time.perf_counter()
        self.current_point_index = 0
        self.current_execution_time = 0.0
        self.execution_log.clear()
        self.execution_completed = False
        
        # Record measurement start time
        self.measurement_start_time = time.time()
        
        # COMPREHENSIVE HARDWARE METADATA LOGGING
        self._log_hardware_metadata()
        
        # Sort points by time and process for execution
        self.sorted_points = sorted(self.path_points, key=lambda p: p.time)
        
        # Debug logging of actual path points being executed
        logger.info(f"Executing path with {len(self.sorted_points)} points:")
        total_duration = 0.0
        for i, point in enumerate(self.sorted_points):
            relative_time = self._get_relative_time(i)
            total_duration = point.time
            logger.info(f"  Point {i}: duration={relative_time:.1f}s, absolute={point.time:.1f}s, freq={point.frequency:.3f}MHz, amp={point.amplitude:.2f}Vpp")
        logger.info(f"Total execution time: {total_duration:.1f}s")
        
        # Save current path design to HDF5
        self._save_path_design_to_hdf5()
        
        # Create and start worker thread
        self.fg_worker = FunctionGeneratorWorker(
            self.function_generator_controller,
            self.sorted_points,  # Pass sorted points, worker will process them
            self.execution_log,
            self.main_window  # Pass main window for HDF5 timeline logging
        )
        
        # Connect worker signals
        self.fg_worker.status_update.connect(self._on_execution_status_update)
        self.fg_worker.execution_finished.connect(self._on_execution_finished)
        
        # Setup live update timer for UI (graph line updates) - reduced frequency for performance
        self.live_update_timer = QTimer()
        self.live_update_timer.timeout.connect(self._update_live_line)
        self.live_update_timer.start(50)  # Update UI every 50ms (20 Hz) for smooth animation
        
        # Update UI
        self.execute_path_btn.setEnabled(False)
        self.stop_execution_btn.setEnabled(True)
        self.status_display.set_status("Starting...")
        
        # Start the worker
        self.fg_worker.start()
        
        logger.info(f"Started force path execution with {len(self.sorted_points)} points")
        self.path_execution_started.emit()
        
    def _on_execution_status_update(self, status_text, frequency, amplitude):
        """Handle status updates from the worker thread."""
        # Always show "Executing" during execution for consistent blue status
        self.status_display.set_status("Executing")
        
    def _on_execution_finished(self, success):
        """Handle execution completion from worker thread."""
        if success:
            self.execution_completed = True
            self.status_display.set_status("Completed")  # Green
            logger.info("Force path execution completed successfully")
        else:
            self.execution_completed = False
            self.status_display.set_status("Error")       # Red
            logger.error("Force path execution failed")
            
        # Clean up execution state
        self._stop_execution()
        
    def _update_live_line(self):
        """Update the vertical line showing current execution position."""
        if not self.is_executing:
            return
            
        # Calculate current execution time
        self.current_execution_time = time.perf_counter() - self.execution_start_time
        
        # Update the live line on the graph
        if hasattr(self, 'ax1') and self.sorted_points:
            # Use processed points to get actual total duration
            processed_points = self._process_points_for_display()
            total_duration = processed_points[-1].time if processed_points else 0
            
            # Remove old live line if it exists
            if self.live_line:
                try:
                    self.live_line.remove()
                except:
                    pass
                    
            # Add new live line if still within execution time
            if self.current_execution_time <= total_duration:
                self.live_line = self.ax1.axvline(
                    x=self.current_execution_time, 
                    color='green', 
                    linewidth=2, 
                    linestyle='--',
                    alpha=0.8,
                    label='Current Position'
                )
                
                # Update canvas
                self.canvas.draw_idle()
                
    def _stop_execution(self):
        """Stop path execution and turn off function generator output."""
        logger.info("Force path execution: Stop requested by user")
        
        # Stop the worker thread
        if hasattr(self, 'fg_worker') and self.fg_worker.isRunning():
            self.fg_worker.stop()
            self.fg_worker.wait(2000)  # Wait up to 2 seconds for thread to finish
            
        # Stop live update timer
        if hasattr(self, 'live_update_timer'):
            self.live_update_timer.stop()
            
        # Remove live line
        if self.live_line:
            try:
                self.live_line.remove()
                self.canvas.draw_idle()
            except:
                pass
            self.live_line = None
            
        # CRITICAL: Turn off function generator output when stopped
        if hasattr(self, 'function_generator_controller') and self.function_generator_controller and self.function_generator_controller.is_connected:
            try:
                logger.info("Force path execution: Turning off function generator output")
                self.function_generator_controller.stop_all_outputs()
            except Exception as e:
                logger.error(f"Force path execution: Failed to turn off output: {e}")
        
        self.is_executing = False
        
        # Save execution results to HDF5
        self._save_execution_to_hdf5()
        
        # Update UI with proper status
        self.execute_path_btn.setEnabled(True)
        self.stop_execution_btn.setEnabled(False)
        
        # Set status based on completion state and function generator availability
        if self.execution_completed:
            self.status_display.set_status("Completed")
        else:
            # Check function generator status when execution stops
            if not self.function_generator_controller or not self.function_generator_controller.is_connected:
                self.status_display.set_status("Disconnected")
            else:
                self.status_display.set_status("Stopped")  # Orange color for user-stopped
        
        completion_status = "Completed" if self.execution_completed else "Stopped"
        logger.info(f"Force path execution {completion_status}")
        self.path_execution_stopped.emit()
        
    def _process_points_for_display(self):
        """Process points to handle time 0 hold behavior for display and execution."""
        if not self.sorted_points:
            return []
            
        processed_points = self.sorted_points.copy()
        
        # Check if first point is not at time 0
        first_point = processed_points[0]
        if first_point.time > 0:
            # Add a hold point from 0 to first point time
            hold_point = PathPoint(
                time=0.0,
                frequency=first_point.frequency,
                amplitude=first_point.amplitude,
                transition=TransitionType.HOLD
            )
            processed_points.insert(0, hold_point)
            
        return processed_points
            
    def _save_path_design_to_hdf5(self):
        """Save current path design to video HDF5 file (consolidated storage)."""
        if not self.main_window or not self.path_points:
            return
        
        # Only save during execution if measurement is active and video recording is active
        if not hasattr(self.main_window, 'measurement_active') or not self.main_window.measurement_active:
            return
            
        # Check if camera widget has active HDF5 recorder
        if not hasattr(self.main_window, 'camera_widget') or not self.main_window.camera_widget:
            return
            
        camera_widget = self.main_window.camera_widget
        if not hasattr(camera_widget, 'hdf5_recorder') or not camera_widget.hdf5_recorder:
            return
            
        hdf5_recorder = camera_widget.hdf5_recorder
        if not hdf5_recorder.is_recording:
            return
            
        try:
            # Prepare path data
            times = [point.time for point in self.path_points]
            frequencies = [point.frequency for point in self.path_points]
            amplitudes = [point.amplitude for point in self.path_points]
            
            execution_data = {
                'path_design_points': len(self.path_points),
                'path_duration_seconds': max(times) if times else 0.0,
                'frequency_range': f"{min(frequencies)}-{max(frequencies)} MHz" if frequencies else "None",
                'amplitude_range': f"{min(amplitudes)}-{max(amplitudes)} Vpp" if amplitudes else "None"
            }
            
            # Log directly to video HDF5 file (consolidated storage)
            hdf5_recorder.log_execution_data('force_path_design', execution_data)
            logger.info("Force path design logged to video HDF5 file")
            
        except Exception as e:
            logger.error(f"Error logging force path design: {e}")
            
    def _save_execution_to_hdf5(self):
        """Log execution results to video HDF5 file (consolidated storage)."""
        if not self.main_window or not self.measurement_start_time:
            return
            
        # Check if camera widget has active HDF5 recorder
        if not hasattr(self.main_window, 'camera_widget') or not self.main_window.camera_widget:
            return
            
        camera_widget = self.main_window.camera_widget
        if not hasattr(camera_widget, 'hdf5_recorder') or not camera_widget.hdf5_recorder:
            return
            
        hdf5_recorder = camera_widget.hdf5_recorder
        if not hdf5_recorder.is_recording:
            return
            
        try:
            execution_data = {
                'execution_completed': self.execution_completed,
                'execution_log_entries': len(self.execution_log),
                'execution_duration_seconds': len(self.execution_log) * 0.02 if self.execution_log else 0  # 50Hz = 20ms intervals
            }
            
            if self.execution_log:
                # Add summary statistics
                set_frequencies = [entry['set_frequency_mhz'] for entry in self.execution_log]
                set_amplitudes = [entry['set_amplitude_vpp'] for entry in self.execution_log]
                
                execution_data.update({
                    'frequency_range_executed': f"{min(set_frequencies)}-{max(set_frequencies)} MHz",
                    'amplitude_range_executed': f"{min(set_amplitudes)}-{max(set_amplitudes)} Vpp",
                    'total_execution_steps': len(set_frequencies)
                })
            
            # Log directly to video HDF5 file (consolidated storage)
            hdf5_recorder.log_execution_data('force_path_execution', execution_data)
            
            status = "completed" if self.execution_completed else "stopped early"
            logger.info(f"Force path execution logged to video HDF5 file: {len(self.execution_log)} steps, {status}")
            
        except Exception as e:
            logger.error(f"Error logging force path execution: {e}")
    
    def _get_unique_filename(self, filename):
        """Generate unique filename by adding _1, _2, etc. if file exists."""
        import os
        
        if not os.path.exists(filename):
            return filename
            
        # Split filename into base and extension
        base, ext = os.path.splitext(filename)
        counter = 1
        
        # Keep incrementing counter until we find a unique filename
        while True:
            new_filename = f"{base}_{counter}{ext}"
            if not os.path.exists(new_filename):
                return new_filename
            counter += 1
        
    def _export_to_hdf5(self):
        """Export force path data and execution log to HDF5 file."""
        from PyQt5.QtWidgets import QFileDialog
        from datetime import datetime
        import h5py
        import numpy as np
        import os
        
        if not self.path_points:
            logger.warning("Export failed: No path points to export")
            return
        
        # Get filename from user
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Force Path to HDF5", 
            f"force_path_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5",
            "HDF5 Files (*.h5 *.hdf5)"
        )
        
        if not filename:
            return
        
        # Handle unique filename generation (add _1, _2, etc. if file exists)
        filename = self._get_unique_filename(filename)
        
        try:
            with h5py.File(filename, 'w') as f:
                # Create main groups
                path_group = f.create_group('force_path')
                execution_group = f.create_group('execution_log')
                fg_timeline_group = f.create_group('function_generator_timeline')  # NEW: Dedicated FG timeline
                metadata_group = f.create_group('hardware_metadata')
                
                # Export path design
                times = [point.time for point in self.path_points]
                frequencies = [point.frequency for point in self.path_points]
                amplitudes = [point.amplitude for point in self.path_points]
                transitions = [point.transition.value for point in self.path_points]
                relative_times = [self._get_relative_time(i) for i in range(len(self.path_points))]
                
                path_group.create_dataset('absolute_times_seconds', data=np.array(times))
                path_group.create_dataset('relative_durations_seconds', data=np.array(relative_times))
                path_group.create_dataset('frequencies_mhz', data=np.array(frequencies))
                path_group.create_dataset('amplitudes_vpp', data=np.array(amplitudes))
                path_group.create_dataset('transitions', data=np.array(transitions, dtype='S10'))
                
                # Export enhanced execution log if available
                if self.execution_log:
                    # Extract all fields from enhanced logging
                    exec_times = [entry.get('execution_time_s', 0) for entry in self.execution_log]
                    abs_timestamps = [entry.get('absolute_timestamp', 0) for entry in self.execution_log]
                    iso_timestamps = [entry.get('iso_timestamp', '') for entry in self.execution_log]
                    set_frequencies = [entry.get('set_frequency_mhz', 0) for entry in self.execution_log]
                    set_amplitudes = [entry.get('set_amplitude_vpp', 0) for entry in self.execution_log]
                    interpolation_progress = [entry.get('interpolation_progress', 0) for entry in self.execution_log]
                    current_segments = [entry.get('current_segment', '') for entry in self.execution_log]
                    transition_types = [entry.get('transition_type', '') for entry in self.execution_log]
                    delta_freqs = [entry.get('delta_freq_mhz', 0) for entry in self.execution_log]
                    delta_amps = [entry.get('delta_amp_vpp', 0) for entry in self.execution_log]
                    
                    # Create execution datasets with comprehensive data
                    execution_group.create_dataset('execution_times_seconds', data=np.array(exec_times))
                    execution_group.create_dataset('absolute_timestamps', data=np.array(abs_timestamps))
                    execution_group.create_dataset('iso_timestamps', data=np.array(iso_timestamps, dtype='S25'))
                    execution_group.create_dataset('set_frequencies_mhz', data=np.array(set_frequencies))
                    execution_group.create_dataset('set_amplitudes_vpp', data=np.array(set_amplitudes))
                    execution_group.create_dataset('interpolation_progress', data=np.array(interpolation_progress))
                    execution_group.create_dataset('current_segments', data=np.array(current_segments, dtype='S10'))
                    execution_group.create_dataset('transition_types', data=np.array(transition_types, dtype='S10'))
                    execution_group.create_dataset('frequency_deltas_mhz', data=np.array(delta_freqs))
                    execution_group.create_dataset('amplitude_deltas_vpp', data=np.array(delta_amps))
                    
                    # NEW: Dedicated Function Generator Timeline Data (for easy video correlation)
                    fg_timeline_group.create_dataset('time_seconds', data=np.array(exec_times))
                    fg_timeline_group.create_dataset('frequency_mhz', data=np.array(set_frequencies))
                    fg_timeline_group.create_dataset('amplitude_vpp', data=np.array(set_amplitudes))
                    fg_timeline_group.create_dataset('unix_timestamps', data=np.array(abs_timestamps))
                    fg_timeline_group.create_dataset('human_timestamps', data=np.array(iso_timestamps, dtype='S25'))
                    
                    # Add descriptive attributes for easy understanding
                    fg_timeline_group.attrs['description'] = 'Function Generator Output Timeline - Real-time values sent to device'
                    fg_timeline_group.attrs['time_reference'] = 'Relative to execution start (time_seconds) and absolute (unix_timestamps)'
                    fg_timeline_group.attrs['frequency_units'] = 'MHz'
                    fg_timeline_group.attrs['amplitude_units'] = 'Vpp (Volts peak-to-peak)'
                    fg_timeline_group.attrs['sample_rate_hz'] = '50 (20ms intervals)'
                    fg_timeline_group.attrs['total_samples'] = len(exec_times)
                    fg_timeline_group.attrs['duration_seconds'] = max(exec_times) if exec_times else 0
                    
                    # Add execution summary statistics
                    execution_group.attrs['total_execution_time_s'] = max(exec_times) if exec_times else 0
                    execution_group.attrs['total_updates'] = len(exec_times)
                    execution_group.attrs['frequency_range_mhz'] = f"{min(set_frequencies):.3f}-{max(set_frequencies):.3f}" if set_frequencies else "0-0"
                    execution_group.attrs['amplitude_range_vpp'] = f"{min(set_amplitudes):.2f}-{max(set_amplitudes):.2f}" if set_amplitudes else "0-0"
                
                # Export comprehensive hardware metadata
                if hasattr(self, 'hardware_metadata'):
                    self._export_metadata_to_hdf5(metadata_group, self.hardware_metadata)
                else:
                    # Fallback basic metadata if detailed metadata not available
                    metadata_group.attrs['export_timestamp'] = time.time()
                    metadata_group.attrs['software_version'] = '1.0.0'
                
                # Add file-level attributes for quick identification
                f.attrs['file_type'] = 'afs_force_path_execution'
                f.attrs['creation_time'] = time.time()
                f.attrs['total_path_points'] = len(self.path_points)
                f.attrs['total_execution_updates'] = len(self.execution_log) if self.execution_log else 0
            
            logger.info(f"Force path exported successfully to: {filename} "
                       f"({len(self.path_points)} points, {len(self.execution_log)} log entries, "
                       f"function generator timeline included)")
            
            # Log the file structure for user reference
            logger.info(f"HDF5 structure: /force_path/, /execution_log/, /function_generator_timeline/, /hardware_metadata/")
            
        except Exception as e:
            logger.error(f"Failed to export force path to HDF5: {e}")
    
    def _export_metadata_to_hdf5(self, metadata_group, metadata):
        """Export comprehensive metadata to HDF5 group."""
        try:
            # Experiment info
            exp_group = metadata_group.create_group('experiment_info')
            exp_info = metadata['experiment_info']
            exp_group.attrs['start_timestamp'] = exp_info['start_timestamp']
            exp_group.attrs['start_iso'] = exp_info['start_iso']
            exp_group.attrs['experiment_type'] = exp_info['experiment_type']
            exp_group.attrs['software_version'] = exp_info['software_version']
            exp_group.attrs['operator'] = exp_info['operator']
            exp_group.attrs['platform'] = exp_info['system_info']['platform']
            exp_group.attrs['python_version'] = exp_info['system_info']['python_version']
            exp_group.attrs['working_directory'] = exp_info['system_info']['working_directory']
            
            # Path design info
            path_group = metadata_group.create_group('path_design')
            path_info = metadata['path_design']
            path_group.attrs['total_points'] = path_info['total_points']
            path_group.attrs['total_duration_s'] = path_info['total_duration_s']
            
            # Function generator metadata
            if 'function_generator' in metadata:
                fg_group = metadata_group.create_group('function_generator')
                fg_info = metadata['function_generator']
                fg_group.attrs['connected'] = fg_info['connected']
                if fg_info['connected']:
                    fg_group.attrs['model'] = fg_info.get('model', 'Unknown')
                    fg_group.attrs['identification'] = fg_info.get('identification', 'Unknown')
                    fg_group.attrs['interface'] = fg_info['communication']['interface']
                    fg_group.attrs['timeout_ms'] = fg_info['communication']['timeout_ms']
                    fg_group.attrs['waveform_type'] = fg_info['initial_settings']['waveform_type']
                    fg_group.attrs['channel'] = fg_info['initial_settings']['channel']
                else:
                    fg_group.attrs['error'] = fg_info.get('error', fg_info.get('reason', 'Unknown'))
            
            # XY Stage metadata
            if 'xy_stage' in metadata:
                stage_group = metadata_group.create_group('xy_stage')
                stage_info = metadata['xy_stage']
                stage_group.attrs['connected'] = stage_info['connected']
                if stage_info['connected'] and 'settings' in stage_info:
                    stage_group.attrs['current_position'] = stage_info['settings']['current_position']
                    stage_group.attrs['movement_enabled'] = stage_info['settings']['movement_enabled']
                else:
                    stage_group.attrs['error'] = stage_info.get('error', stage_info.get('reason', 'Unknown'))
            
        except Exception as e:
            logger.error(f"Failed to export metadata to HDF5: {e}")
    
    def _log_hardware_metadata(self):
        """Log comprehensive hardware metadata for experiment reproducibility."""
        import datetime
        import platform
        import sys
        import os
        
        metadata = {
            'experiment_info': {
                'start_timestamp': self.measurement_start_time,
                'start_iso': datetime.datetime.fromtimestamp(self.measurement_start_time).isoformat(),
                'experiment_type': 'force_path_execution',
                'software_version': '1.0.0',  # Could be read from version file
                'operator': os.getenv('USERNAME', 'unknown'),
                'system_info': {
                    'platform': platform.platform(),
                    'python_version': sys.version,
                    'working_directory': os.getcwd()
                }
            },
            'path_design': {
                'total_points': len(self.path_points),
                'total_duration_s': max(point.time for point in self.path_points) if self.path_points else 0,
                'path_points': [
                    {
                        'index': i,
                        'absolute_time_s': point.time,
                        'relative_duration_s': self._get_relative_time(i),
                        'frequency_mhz': point.frequency,
                        'amplitude_vpp': point.amplitude,
                        'transition_type': point.transition.value
                    }
                    for i, point in enumerate(self.path_points)
                ]
            }
        }
        
        # Function Generator Settings
        if self.function_generator_controller and self.function_generator_controller.is_connected:
            try:
                # Get function generator identification and settings
                fg_metadata = {
                    'connected': True,
                    'model': getattr(self.function_generator_controller, 'model', 'Siglent SDG1032X'),
                    'identification': 'Siglent SDG1032X',  # Could query from device
                    'communication': {
                        'interface': 'USB-VISA',
                        'timeout_ms': getattr(self.function_generator_controller.function_generator, 'timeout', 5000) if hasattr(self.function_generator_controller, 'function_generator') else 5000,
                        'connection_timestamp': self.measurement_start_time
                    },
                    'initial_settings': {
                        'output_enabled': False,  # Will be enabled during execution
                        'waveform_type': 'sine',
                        'channel': 1
                    }
                }
                metadata['function_generator'] = fg_metadata
            except Exception as e:
                metadata['function_generator'] = {'connected': False, 'error': str(e)}
        else:
            metadata['function_generator'] = {'connected': False, 'reason': 'not_available'}
        
        # Stage Settings (if available through main window)
        if self.main_window and hasattr(self.main_window, 'measurement_controls_widget'):
            try:
                stage_metadata = {
                    'connected': True,
                    'type': 'xy_stage',
                    'settings': {
                        'current_position': 'unknown',  # Could query actual position
                        'movement_enabled': True
                    }
                }
                metadata['xy_stage'] = stage_metadata
            except Exception as e:
                metadata['xy_stage'] = {'connected': False, 'error': str(e)}
        else:
            metadata['xy_stage'] = {'connected': False, 'reason': 'not_available'}
        
        # Store metadata for HDF5 export
        self.hardware_metadata = metadata
        
        # Log key information to console for immediate feedback
        logger.info(f"=== EXPERIMENT METADATA ===")
        logger.info(f"Start Time: {metadata['experiment_info']['start_iso']}")
        logger.info(f"Path Points: {metadata['path_design']['total_points']}")
        logger.info(f"Total Duration: {metadata['path_design']['total_duration_s']:.1f}s")
        logger.info(f"Function Generator: {'Connected' if metadata['function_generator']['connected'] else 'Disconnected'}")
        logger.info(f"XY Stage: {'Connected' if metadata['xy_stage']['connected'] else 'Disconnected'}")
        logger.info(f"===========================")
        
    def set_function_generator_controller(self, controller):
        """Set the function generator controller reference."""
        self.function_generator_controller = controller
        # Update status display based on function generator availability
        self._update_status_display()
        
    def _update_status_display(self):
        """Update status display based on function generator availability."""
        if not self.function_generator_controller or not self.function_generator_controller.is_connected:
            self.status_display.set_status("Disconnected")
        else:
            self.status_display.set_status("Ready")
        
    def set_main_window(self, main_window):
        """Set the main window reference for measurement-driven logging."""
        self.main_window = main_window
        logger.info("Force Path Designer linked to main window for measurement logging")
        
    def get_current_path(self) -> List[PathPoint]:
        """Get the current path points."""
        return self.path_points.copy()
        
    def closeEvent(self, event):
        """Handle widget close event."""
        if self.is_executing:
            self._stop_execution()
        super().closeEvent(event)


class ForcePathDesignerWindow(QMainWindow):
    """Main window wrapper for the Force Path Designer widget."""
    
    # Forward signals from the widget
    path_loaded = pyqtSignal(list)
    path_execution_started = pyqtSignal()
    path_execution_stopped = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Create the central widget
        self.designer_widget = ForcePathDesignerWidget(self)
        self.setCentralWidget(self.designer_widget)
        
        # Forward signals
        self.designer_widget.path_loaded.connect(self.path_loaded.emit)
        self.designer_widget.path_execution_started.connect(self.path_execution_started.emit)
        self.designer_widget.path_execution_stopped.connect(self.path_execution_stopped.emit)
        
        # Set window properties
        self.setWindowTitle("Force Path Designer")
        self.resize(1100, 500)  # Wider for better table and graph visibility
        
        # Center the window on screen
        self._center_on_screen()
        
    def _center_on_screen(self):
        """Center the window on the screen."""
        from PyQt5.QtWidgets import QApplication
        screen = QApplication.desktop().screenGeometry()
        size = self.geometry()
        self.move(
            (screen.width() - size.width()) // 2,
            (screen.height() - size.height()) // 2
        )
        
    def set_function_generator_controller(self, controller):
        """Set the function generator controller reference."""
        self.designer_widget.set_function_generator_controller(controller)
        
    def get_current_path(self) -> List[PathPoint]:
        """Get the current path points."""
        return self.designer_widget.get_current_path()
        
    def closeEvent(self, event):
        """Handle window close event."""
        self.designer_widget.closeEvent(event)
        super().closeEvent(event)


def main():
    """Test the force path designer widget."""
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    window = ForcePathDesignerWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()