"""BlackFiber MacBook-side application code.

Modules:
    config           - Centralized configuration loaded from .env
    engagement       - DroneEngagement dataclass (matches Foundry ontology)
    state_machine    - EngagementState enum + transition logic
    logger           - LocalJSONLogger + FoundryOSDKLogger (same interface)
    calibration      - Pixel <-> servo angle calibration (IDW interpolation)
    detector         - OpenCV motion detection + YOLO ensemble
    overlay          - Status overlay drawn on camera frames
    serial_link      - Pico W serial bridge (with --mock for hardware-less dev)
    calibration_tool - Manual jog + click-to-aim teleop UI
    main_tracker     - Live pipeline glue
"""
