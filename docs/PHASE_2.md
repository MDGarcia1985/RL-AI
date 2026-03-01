**Future / target layout** for rc_guardian (hardware integration and UGV stack). The current NNABL_POC project lives under `NNABL_POC/rc_agents/`; see `docs/PHASE_1.md` for that structure.

rc_guardian/
├─ README.md
├─ pyproject.toml                 # python deps + tooling (ruff/black/pytest)
├─ pytest.ini                     # test discovery + markers
├─ .gitignore
│
├─ docs/
│  ├─ wiring.md
│  ├─ safety_rules.md
│  ├─ bringup_checklist.md
│  └─ demo_script.md
│
├─ common/
│  ├─ protocol/
│  │  ├─ messages.yaml            # DriveCommand, SafetyEvent, VisionEvent, GpsFix...
│  │  ├─ protocol_version.txt
│  │  ├─ framing.md               # start bytes, length, CRC16, retry behavior
│  │  ├─ message_ids.py
│  │  ├─ message_ids.h
│  │  ├─ codec/
│  │  │  ├─ codec_py.py
│  │  │  └─ codec_cpp.h
│  │  └─ tests/
│  │     ├─ test_ids_unique.py
│  │     └─ test_codec_roundtrip.py
│  │
│  ├─ config/
│  │  └─ vehicle.yaml             # thresholds, limits, PID, etc.
│  │
│  ├─ hw/
│  │  ├─ pinmap.yaml
│  │  └─ wiring_diagram.md
│  │
│  └─ safety/
│     ├─ fault_codes.yaml
│     ├─ safety_states.yaml
│     └─ tests/
│        └─ test_fault_codes_unique.py
│
├─ firmware/
│  └─ unoq_mcu/
│     ├─ platformio.ini
│     ├─ include/
│     │  └─ protocol_codec.hpp
│     ├─ src/
│     │  ├─ main.cpp
│     │  ├─ control/
│     │  │  ├─ motor_control.cpp
│     │  │  └─ steering_servo.cpp
│     │  ├─ sensors/
│     │  │  ├─ bumper.cpp
│     │  │  ├─ tof_vl53l0x.cpp
│     │  │  └─ i2c_mux.cpp
│     │  ├─ safety/
│     │  │  ├─ estop.cpp
│     │  │  └─ watchdog.cpp
│     │  └─ bridge/
│     │     ├─ rpc_endpoints.cpp
│     │     └─ protocol_codec.cpp
│     └─ test/                    # PlatformIO native/embedded tests
│        ├─ test_estop.cpp
│        └─ test_watchdog.cpp
│
├─ edge_ai/                       # runs on UNO Q MPU (Linux)
│  ├─ pyproject.toml              # optional if you deploy separately
│  ├─ rcg_edge/
│  │  ├─ __init__.py
│  │  ├─ main.py                  # MPU entrypoint
│  │  ├─ vision/
│  │  │  ├─ oakd_pipeline.py
│  │  │  ├─ detectors.py
│  │  │  └─ depth_to_events.py
│  │  ├─ fusion/
│  │  │  ├─ perception_fusion.py
│  │  │  ├─ gps_localization.py
│  │  │  └─ filters.py
│  │  ├─ bridge/
│  │  │  ├─ mpu_to_mcu.py
│  │  │  └─ mcu_to_mpu.py
│  │  └─ agents/
│  │     ├─ manual_agent.py
│  │     ├─ rule_agent.py
│  │     └─ rl_agent.py
│  └─ tests/
│     ├─ test_fusion_filters.py
│     └─ test_vision_event_mapping.py
│
├─ host/                          # sim + dev tooling (PC)
│  ├─ rcg_host/
│  │  ├─ __init__.py
│  │  ├─ main.py                  # host entrypoint (cli)
│  │  ├─ env/
│  │  │  ├─ grid_env.py
│  │  │  └─ rc_env.py
│  │  ├─ controller/
│  │  │  └─ bot_controller.py
│  │  ├─ logging/
│  │  │  ├─ movement_logger.py
│  │  │  └─ telemetry_logger.py
│  │  └─ comms/
│  │     ├─ serial_transport.py
│  │     └─ protocol.py
│  └─ tests/
│     ├─ test_grid_env.py
│     └─ test_logger_writes.py
│
├─ ui/
│  └─ tk/
│     ├─ rcg_ui/
│     │  ├─ __init__.py
│     │  ├─ main.py               # UI entrypoint (optional)
│     │  ├─ gui_main.py
│     │  └─ widgets/
│     │     ├─ dpad.py
│     │     └─ telemetry_panel.py
│     └─ tests/
│        └─ test_ui_smoke.py       # basic import/run tests
│
├─ tools/
│  ├─ tof_calibration/
│  ├─ steering_center/
│  ├─ motor_sweep/
│  ├─ gps_calibration/
│  ├─ log_replay/
│  └─ hil/                        # hardware-in-the-loop scripts
│     ├─ estop_test.py
│     ├─ tof_sweep_test.py
│     └─ latency_benchmark.py
│
├─ data/
│  ├─ botmovements.txt            # existing move log
│  ├─ waypoints/
│  │  ├─ patrol_routes.gpx
│  │  └─ boundary_coords.json
│  └─ runs/
│     └─ 2026-02-01_run_001/
│        ├─ telemetry.log
│        ├─ vision_events.log
│        ├─ gps_track.gpx
│        └─ notes.md
│
└─ tests/                         # root integration tests only
   ├─ test_protocol_py_cpp_consistency.py
   ├─ test_safety_gate_policy.py
   └─ test_end_to_end_manual_mode.py