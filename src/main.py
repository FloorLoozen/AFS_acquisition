"""Main application entry point for AFS Tracking System."""

import sys
import os
from typing import NoReturn


def setup_python_path() -> None:
    """Setup Python path for module imports."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def create_application() -> 'QApplication':
    """Create and configure the PyQt5 application with optimized settings."""
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import Qt
    
    # Enable high DPI support for better display on modern monitors
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    
    # Set application metadata for better OS integration
    app.setApplicationName("AFS Tracking System")
    app.setApplicationVersion("3.0")
    app.setOrganizationName("AFS Lab")
    
    return app


def pre_initialize_hardware():
    """Pre-initialize hardware controllers for fast runtime access."""
    from src.utils.logger import get_logger
    logger = get_logger("hardware_init")
    
    logger.info("Pre-initializing hardware controllers...")
    
    # Pre-load and initialize hardware controllers
    hardware_status = {}
    
    try:
        # Initialize stage manager
        from src.controllers.stage_manager import StageManager
        stage_manager = StageManager.get_instance()
        # Don't connect yet, just pre-load the module
        hardware_status['stage_manager'] = 'loaded'
        
        # Pre-load camera controller module
        from src.controllers.camera_controller import CameraController
        hardware_status['camera_controller'] = 'loaded'
        
        # Pre-load function generator controller
        from src.controllers.function_generator_controller import FunctionGeneratorController
        hardware_status['function_generator'] = 'loaded'
        
        logger.info(f"Hardware modules pre-loaded: {list(hardware_status.keys())}")
        return hardware_status
        
    except Exception as e:
        logger.warning(f"Hardware pre-loading error (non-critical): {e}")
        return {}


def main() -> NoReturn:
    """Initialize and run the AFS Tracking application.
    
    Hybrid approach: Eager hardware initialization for performance,
    lazy UI imports to minimize startup memory usage.
    
    Raises:
        SystemExit: Always exits with the application return code.
    """
    # Setup environment
    setup_python_path()
    
    # Initialize logging first (lightweight)
    from src.utils.logger import get_logger
    logger = get_logger("main")
    
    try:
        # Initialize configuration and performance monitoring
        from src.utils.config_manager import get_config, auto_optimize_config
        from src.utils.performance_monitor import get_performance_monitor, start_monitoring, measure_time
        
        # Auto-optimize configuration based on system capabilities
        config = get_config()
        auto_optimize_config()
        
        # Start performance monitoring if enabled
        if config.performance.enable_performance_monitoring:
            start_monitoring()
            logger.info("Performance monitoring enabled")
        
        # Pre-initialize hardware for fast access during runtime
        with measure_time("hardware_pre_init"):
            hardware_status = pre_initialize_hardware()
        
        # Create application with optimized settings
        with measure_time("app_creation"):
            app = create_application()
        
        # Import and create main window with timing
        with measure_time("main_window_creation"):
            from src.ui.main_window import MainWindow
            window = MainWindow()
        
        # Start in maximized mode for better camera view (don't restore geometry to keep maximized)
        window.showMaximized()
        logger.info("Application started in maximized mode")
        
        # Start event loop
        logger.info("AFS Tracking System started successfully")
        logger.info(f"Hardware modules ready: {list(hardware_status.keys())}")
        
        rc = app.exec_()
        
        # Save configuration on exit
        config.save_config()
        
        # Stop monitoring and show performance summary
        from src.utils.performance_monitor import stop_monitoring, get_performance_report
        if config.performance.enable_performance_monitoring:
            stop_monitoring()
            # Log performance summary
            performance_summary = get_performance_monitor().get_performance_summary()
            if performance_summary.get('recommendations'):
                logger.info(f"Performance recommendations: {performance_summary['recommendations']}")
        
        logger.info(f"Application exited with code {rc}")
        sys.exit(rc)
        
    except ImportError as e:
        error_msg = f"Import error: {e}\nPlease ensure all dependencies are installed: pip install -r requirements.txt"
        try:
            logger.error(error_msg)
        except:
            print(error_msg)
        sys.exit(1)
    except Exception as e:
        error_msg = f"Critical error during startup: {e}"
        try:
            logger.error(error_msg)
            # Also print to console in case logging fails
            print(error_msg)
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
        except:
            print(error_msg)
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()