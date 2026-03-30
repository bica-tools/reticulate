"""Synchronous parallel benchmark protocols — realistic encodings.

Real-world protocols where synchronous parallelism (⊗) is mandatory.
Each voice type reflects actual signal/method names from specifications,
datasheets, or standards documents.

22 benchmarks across 8 industrial domains.
"""

# ===================================================================
# 1. Manufacturing & Assembly
# ===================================================================

AUTOMOTIVE_WELDING = {
    "name": "Automotive Body Welding (KUKA KR C4)",
    "description": "Left/right weld robots on a BIW (body-in-white) station. "
                   "Robots share a zone interlock and fire weld guns in sync. "
                   "Based on KUKA KR C4 controller sequencing.",
    "voices": [
        # Left robot — 8 welds per cycle, 3 retries allowed
        "rec X . &{homePos: &{zoneRequest: +{GRANTED: &{approach: &{clamp: &{weldPulse: +{qualityOK: &{unclamp: &{retract: &{zoneRelease: X}}}, qualityNOK: +{retry: &{weldPulse: +{qualityOK: &{unclamp: &{retract: &{zoneRelease: X}}}, qualityNOK: &{markDefect: &{unclamp: &{retract: &{zoneRelease: X}}}}}}, scrap: &{unclamp: &{retract: &{zoneRelease: X}}}}}}}}, DENIED: &{waitZone: X}}, eStop: end}}",
        # Right robot — symmetric protocol
        "rec X . &{homePos: &{zoneRequest: +{GRANTED: &{approach: &{clamp: &{weldPulse: +{qualityOK: &{unclamp: &{retract: &{zoneRelease: X}}}, qualityNOK: +{retry: &{weldPulse: +{qualityOK: &{unclamp: &{retract: &{zoneRelease: X}}}, qualityNOK: &{markDefect: &{unclamp: &{retract: &{zoneRelease: X}}}}}}, scrap: &{unclamp: &{retract: &{zoneRelease: X}}}}}}}}, DENIED: &{waitZone: X}}, eStop: end}}",
    ],
    "why_sync": "Zone interlock: both robots must request/release zone together; "
                "weld guns fire on shared timer tick; frame distorts under asymmetric heat",
}

BOTTLING_PLANT = {
    "name": "Bottling Plant (Krones Modulfill)",
    "description": "Filler-capper-labeler on single conveyor. Each station "
                   "has sensors, actuators, and fault recovery. "
                   "Based on Krones Modulfill IPC protocol.",
    "voices": [
        # Filler valve (volumetric dosing)
        "rec X . &{bottleDetect: &{valveOpen: &{flowMeter: +{targetReached: &{valveClose: &{drainCheck: +{levelOK: X, overfill: &{purge: X}}}}, sensorFault: &{valveClose: &{rejectBottle: X}}}}}, lineStop: end}",
        # Capper head (pneumatic torque)
        "rec X . &{bottleDetect: &{capFeed: +{capPresent: &{descend: &{torqueApply: +{torqueOK: &{ascend: X}, torqueLow: &{ascend: &{rejectBottle: X}}, torqueHigh: &{ascend: &{rejectBottle: X}}}}}, capMissing: &{rejectBottle: X}}}, lineStop: end}",
        # Labeler (servo-driven)
        "rec X . &{bottleDetect: &{labelFeed: &{vacuumGrip: +{labelReady: &{applyLabel: &{pressRoll: +{scanOK: X, scanFail: &{rejectBottle: X}}}}, feedJam: &{cutLabel: &{rejectBottle: X}}}}}, lineStop: end}",
    ],
    "why_sync": "Conveyor indexes all three stations simultaneously on bottleDetect; "
                "capper can't descend before filler finishes on same bottle",
}

SEMICONDUCTOR_LITHO = {
    "name": "Dual-Beam EUV Lithography (ASML Twinscan)",
    "description": "Two exposure units share wafer stage. "
                   "Alignment must be nanometer-precise across beams.",
    "voices": [
        # Beam A (13.5nm EUV source)
        "rec X . &{waferLoad: &{globalAlign: &{fineAlign: &{expose: &{measureOverlay: +{overlayOK: X, overlayShift: &{reExpose: X}}}}}}, lotComplete: end}",
        # Beam B (13.5nm EUV source)
        "rec X . &{waferLoad: &{globalAlign: &{fineAlign: &{expose: &{measureOverlay: +{overlayOK: X, overlayShift: &{reExpose: X}}}}}}, lotComplete: end}",
    ],
    "why_sync": "Overlay error budget is 1.5nm; asynchronous exposure accumulates "
                "thermal drift exceeding budget in <10 seconds",
}


# ===================================================================
# 2. Avionics & Safety-Critical
# ===================================================================

TRIPLE_MODULAR_REDUNDANCY = {
    "name": "TMR Flight Computer (DO-178C DAL A)",
    "description": "Three-channel FCC with bitwise voter. "
                   "Based on Airbus A320 ELAC architecture. "
                   "Channels compute same function; voter checks 2-of-3.",
    "voices": [
        # Channel A (COM)
        "rec X . &{adcSample: &{imuRead: &{sensorFusion: &{lawCompute: &{actuatorCmd: &{voterSubmit: +{twoOfThree: X, disagree: &{selfTest: +{healthy: X, failed: &{engagePassive: end}}}}}}}}}}",
        # Channel B (MON)
        "rec X . &{adcSample: &{imuRead: &{sensorFusion: &{lawCompute: &{actuatorCmd: &{voterSubmit: +{twoOfThree: X, disagree: &{selfTest: +{healthy: X, failed: &{engagePassive: end}}}}}}}}}}",
        # Channel C (STBY)
        "rec X . &{adcSample: &{imuRead: &{sensorFusion: &{lawCompute: &{actuatorCmd: &{voterSubmit: +{twoOfThree: X, disagree: &{selfTest: +{healthy: X, failed: &{engagePassive: end}}}}}}}}}}",
    ],
    "why_sync": "Voter window is 2ms; if channel B is 1 cycle behind, voter compares "
                "current-frame A with stale-frame B → false disagree → nuisance disconnect",
}

FLY_BY_WIRE = {
    "name": "Fly-by-Wire Aileron Actuators (A320 ELAC)",
    "description": "Left/right aileron servos driven by same ELAC output. "
                   "Differential position >0.5° triggers ECAM caution.",
    "voices": [
        # Left aileron servo
        "rec X . &{demandReceive: &{positionRead: &{errorCompute: &{servoCommand: &{positionFeedback: +{inBand: X, outOfBand: &{rateLimitApply: X}, servoFault: &{dampenMode: end}}}}}}}",
        # Right aileron servo
        "rec X . &{demandReceive: &{positionRead: &{errorCompute: &{servoCommand: &{positionFeedback: +{inBand: X, outOfBand: &{rateLimitApply: X}, servoFault: &{dampenMode: end}}}}}}}",
    ],
    "why_sync": "80ms frame time; one servo lagging one frame = 2° split at max rate "
                "→ uncommanded roll → ECAM alert → pilot intervention",
}

ABS_BRAKING = {
    "name": "ABS/ESC Hydraulic Unit (Bosch ESP 9.3)",
    "description": "Four wheel speed sensors + four solenoid valves. "
                   "Based on Bosch ESP 9.3 with 4ms control cycle.",
    "voices": [
        # Front-left channel
        "rec X . &{wheelSpeedRead: &{slipCalc: +{noSlip: &{holdPressure: X}, slipDetect: &{releasePressure: &{waitRecover: &{reapplyPressure: X}}}, sensorFault: &{fallbackABS: end}}}}",
        # Front-right channel
        "rec X . &{wheelSpeedRead: &{slipCalc: +{noSlip: &{holdPressure: X}, slipDetect: &{releasePressure: &{waitRecover: &{reapplyPressure: X}}}, sensorFault: &{fallbackABS: end}}}}",
        # Rear-left channel
        "rec X . &{wheelSpeedRead: &{slipCalc: +{noSlip: &{holdPressure: X}, slipDetect: &{releasePressure: &{waitRecover: &{reapplyPressure: X}}}, sensorFault: &{fallbackABS: end}}}}",
        # Rear-right channel
        "rec X . &{wheelSpeedRead: &{slipCalc: +{noSlip: &{holdPressure: X}, slipDetect: &{releasePressure: &{waitRecover: &{reapplyPressure: X}}}, sensorFault: &{fallbackABS: end}}}}",
    ],
    "why_sync": "4ms cycle; diagonal brake bias requires FL+RR and FR+RL to pulse "
                "within same cycle; 8ms skew → yaw moment → loss of directional control",
}


# ===================================================================
# 3. Telecommunications
# ===================================================================

TDM_MULTIPLEXING = {
    "name": "E1/T1 TDM Frame (ITU-T G.704)",
    "description": "32 timeslots per 125μs frame. Each channel gets "
                   "exactly one slot per frame. TS0 carries framing pattern.",
    "voices": [
        # Timeslot 1 (64 kbit/s voice channel)
        "rec X . &{frameSync: &{slotReceive: &{pcmDecode: &{playoutBuffer: X}}}, losDetect: end}",
        # Timeslot 2
        "rec X . &{frameSync: &{slotReceive: &{pcmDecode: &{playoutBuffer: X}}}, losDetect: end}",
        # Timeslot 16 (signaling)
        "rec X . &{frameSync: &{slotReceive: &{casExtract: +{onHook: X, offHook: X, digit: X}}}, losDetect: end}",
        # Timeslot 31
        "rec X . &{frameSync: &{slotReceive: &{pcmDecode: &{playoutBuffer: X}}}, losDetect: end}",
    ],
    "why_sync": "Timeslot boundaries are defined by 8kHz frame clock; "
                "channel N's data occupies bits 8N to 8N+7 of each frame — no flexibility",
}

BEAMFORMING_5G = {
    "name": "5G NR Massive MIMO Beamforming (3GPP TS 38.214)",
    "description": "64-element antenna array; each element has independent "
                   "phase shifter and PA. Beam pattern requires <1° phase coherence.",
    "voices": [
        # Antenna element 0
        "rec X . &{csiReport: &{beamSelect: &{phaseShift: &{paGain: &{ofdmSymbol: +{ackReceived: X, nackReceived: &{retransmit: X}, dtxTimeout: X}}}}}, rfShutdown: end}",
        # Antenna element 1
        "rec X . &{csiReport: &{beamSelect: &{phaseShift: &{paGain: &{ofdmSymbol: +{ackReceived: X, nackReceived: &{retransmit: X}, dtxTimeout: X}}}}}, rfShutdown: end}",
    ],
    "why_sync": "OFDM symbol duration is 66.7μs (15kHz SCS); all elements must "
                "transmit within CP (cyclic prefix) window or ISI destroys subcarriers",
}


# ===================================================================
# 4. Robotics
# ===================================================================

DUAL_ARM_MANIPULATION = {
    "name": "Dual-Arm Cooperative Manipulation (ABB YuMi)",
    "description": "Two 7-DOF arms holding single rigid body (PCB panel). "
                   "Based on ABB YuMi IRB 14000 coordinated motion.",
    "voices": [
        # Left arm (7 joints)
        "rec X . &{pathPlan: &{invKinematics: &{jointInterpolate: &{servoUpdate: &{forceRead: +{forceOK: X, forceExceed: &{compliance: X}, collision: &{safeStop: end}}}}}}}",
        # Right arm (7 joints)
        "rec X . &{pathPlan: &{invKinematics: &{jointInterpolate: &{servoUpdate: &{forceRead: +{forceOK: X, forceExceed: &{compliance: X}, collision: &{safeStop: end}}}}}}}",
    ],
    "why_sync": "1ms servo cycle; 2ms path deviation on rigid body → internal stress "
                "exceeds material yield strength → part snaps",
}

HUMANOID_WALKING = {
    "name": "Humanoid Bipedal Gait (Boston Dynamics Atlas ZMP)",
    "description": "Left/right leg with zero-moment-point (ZMP) balance. "
                   "Gait phases are complementary: left stance = right swing.",
    "voices": [
        # Left leg (hip + knee + ankle)
        "rec X . &{stancePhase: &{midStance: &{terminalStance: &{preSwing: &{initialSwing: &{midSwing: &{terminalSwing: &{initialContact: X}}}}}}}, standStill: end}",
        # Right leg (180° phase offset)
        "rec X . &{initialSwing: &{midSwing: &{terminalSwing: &{initialContact: &{stancePhase: &{midStance: &{terminalStance: &{preSwing: X}}}}}}}, standStill: end}",
    ],
    "why_sync": "ZMP must stay within support polygon at every instant; async leg motion "
                "moves ZMP outside polygon → fall in <200ms",
}

DRONE_SWARM_FORMATION = {
    "name": "Drone Swarm Formation (PX4 Offboard Mode)",
    "description": "Three quadrotors maintaining triangular formation. "
                   "Using PX4 offboard setpoint protocol at 50Hz.",
    "voices": [
        # Leader drone
        "rec X . &{gpsRead: &{imuFuse: &{waypointCalc: &{setpointPublish: &{heartbeat: +{followersOK: X, followerLost: &{holdPosition: X}, geoFence: &{returnToLaunch: end}}}}}}}",
        # Follower 1 (offset +5m east)
        "rec X . &{gpsRead: &{imuFuse: &{leaderTrack: &{offsetCalc: &{setpointPublish: &{heartbeat: +{leaderOK: X, leaderLost: &{holdPosition: X}, geoFence: &{returnToLaunch: end}}}}}}}}",
        # Follower 2 (offset +5m north)
        "rec X . &{gpsRead: &{imuFuse: &{leaderTrack: &{offsetCalc: &{setpointPublish: &{heartbeat: +{leaderOK: X, leaderLost: &{holdPosition: X}, geoFence: &{returnToLaunch: end}}}}}}}}",
    ],
    "why_sync": "50Hz control loop; 3 missed heartbeats (60ms) triggers follower-lost "
                "failsafe; formation spacing tolerance is ±1m at 10m/s",
}


# ===================================================================
# 5. Database & Distributed Systems
# ===================================================================

TWO_PHASE_COMMIT = {
    "name": "XA Two-Phase Commit (JTA/XA spec)",
    "description": "Distributed transaction across two resource managers. "
                   "Transaction manager coordinates prepare→commit/rollback.",
    "voices": [
        # Resource Manager A (e.g., PostgreSQL)
        "&{xaStart: &{executeSQL: &{xaEnd: &{xaPrepare: +{xaOK: &{xaCommit: end}, xaRB: &{xaRollback: end}, rmFail: &{xaRollback: end}}}}}}",
        # Resource Manager B (e.g., Oracle)
        "&{xaStart: &{executeSQL: &{xaEnd: &{xaPrepare: +{xaOK: &{xaCommit: end}, xaRB: &{xaRollback: end}, rmFail: &{xaRollback: end}}}}}}",
    ],
    "why_sync": "If RM-A commits before RM-B prepares, and RM-B votes rollback, "
                "RM-A cannot undo → heuristic hazard → data inconsistency",
}

RAFT_LOG_REPLICATION = {
    "name": "Raft Consensus (etcd/Consul)",
    "description": "Three-node Raft cluster. Leader appends, followers replicate. "
                   "Based on etcd raft implementation.",
    "voices": [
        # Node 1 (current leader)
        "rec X . &{clientRequest: &{appendLog: &{replicateToFollowers: +{majorityAck: &{commitEntry: &{applyToStateMachine: &{clientResponse: X}}}, timeout: &{stepDown: end}}}}}",
        # Node 2 (follower)
        "rec X . &{appendRPC: &{logCheck: +{logMatch: &{appendEntry: &{sendAck: X}}, logConflict: &{truncateLog: &{appendEntry: &{sendAck: X}}}}}, electionTimeout: end}",
        # Node 3 (follower)
        "rec X . &{appendRPC: &{logCheck: +{logMatch: &{appendEntry: &{sendAck: X}}, logConflict: &{truncateLog: &{appendEntry: &{sendAck: X}}}}}, electionTimeout: end}",
    ],
    "why_sync": "CommitIndex must never advance beyond minority-replicated entries; "
                "if leader commits entry N before follower has entry N-1, log diverges "
                "→ split brain on leader election",
}

MPI_BARRIER = {
    "name": "MPI_Barrier + MPI_Allreduce (MPI-3.1 §5.3)",
    "description": "Four-process parallel computation with barrier sync "
                   "and collective reduction. Based on OpenMPI tree algorithm.",
    "voices": [
        # Rank 0
        "rec X . &{localCompute: &{allreduceContrib: &{barrierArrive: &{barrierDepart: &{globalResult: X}}}}, mpiFinalize: end}",
        # Rank 1
        "rec X . &{localCompute: &{allreduceContrib: &{barrierArrive: &{barrierDepart: &{globalResult: X}}}}, mpiFinalize: end}",
        # Rank 2
        "rec X . &{localCompute: &{allreduceContrib: &{barrierArrive: &{barrierDepart: &{globalResult: X}}}}, mpiFinalize: end}",
        # Rank 3
        "rec X . &{localCompute: &{allreduceContrib: &{barrierArrive: &{barrierDepart: &{globalResult: X}}}}, mpiFinalize: end}",
    ],
    "why_sync": "MPI_Barrier semantics: no process may exit barrier until all have "
                "arrived; allreduce result undefined until all contributions received",
}


# ===================================================================
# 6. Digital Hardware
# ===================================================================

SYNCHRONOUS_CIRCUIT = {
    "name": "Pipelined RISC-V Core (3-stage: IF/ID/EX)",
    "description": "Three pipeline stages clocked by single domain clock. "
                   "Based on RISC-V RV32I micro-architecture.",
    "voices": [
        # Instruction Fetch stage
        "rec X . &{clockRise: &{pcRead: &{imemFetch: &{irLatch: &{clockFall: X}}}}, reset: end}",
        # Instruction Decode stage
        "rec X . &{clockRise: &{irRead: &{regFileRead: &{immDecode: &{controlGen: &{clockFall: X}}}}}, reset: end}",
        # Execute stage
        "rec X . &{clockRise: &{aluCompute: &{branchResolve: +{branchTaken: &{flushPipe: &{clockFall: X}}, branchNotTaken: &{wbLatch: &{clockFall: X}}}}}, reset: end}",
    ],
    "why_sync": "Setup/hold time violations: if ID stage latches before IF has "
                "stabilized, decoder reads partial instruction → wrong opcode → "
                "silent data corruption",
}

DDR_MEMORY = {
    "name": "DDR4 SDRAM Read Burst (JEDEC JESD79-4C)",
    "description": "Command, address, and data buses on DDR4 interface. "
                   "CAS latency = CL14 at 3200 MT/s.",
    "voices": [
        # Command/Address bus
        "rec X . &{clockRise: &{csAssert: &{rasNeg: &{casNeg: &{weNeg: +{readCmd: &{nopWait: X}, writeCmd: &{nopWait: X}, refreshCmd: &{nopWait: X}}}}}}, deselect: end}",
        # Data bus (DQ pins)
        "rec X . &{clockRise: &{dqsDrive: &{dqCapture: &{burstBeat: +{beatContinue: X, burstComplete: &{dqsTristate: X}}}}}, deselect: end}",
    ],
    "why_sync": "DQS strobe must be centered on DQ data eye; 150ps timing margin "
                "at 3200 MT/s; any clock domain crossing would violate JEDEC timing",
}

PCIE_LANES = {
    "name": "PCIe Gen4 x4 Link (PCI-SIG ECN)",
    "description": "Four lanes at 16 GT/s with 128b/130b encoding. "
                   "Lane-to-lane skew must be <8 UI.",
    "voices": [
        # Lane 0
        "rec X . &{symbolEncode: &{scramble: &{serializeNRZ: &{transmit: +{ackDllp: X, nakDllp: &{replay: X}, timeout: &{linkRetrain: X}}}}}, ltssm: end}",
        # Lane 1
        "rec X . &{symbolEncode: &{scramble: &{serializeNRZ: &{transmit: +{ackDllp: X, nakDllp: &{replay: X}, timeout: &{linkRetrain: X}}}}}, ltssm: end}",
        # Lane 2
        "rec X . &{symbolEncode: &{scramble: &{serializeNRZ: &{transmit: +{ackDllp: X, nakDllp: &{replay: X}, timeout: &{linkRetrain: X}}}}}, ltssm: end}",
        # Lane 3
        "rec X . &{symbolEncode: &{scramble: &{serializeNRZ: &{transmit: +{ackDllp: X, nakDllp: &{replay: X}, timeout: &{linkRetrain: X}}}}}, ltssm: end}",
    ],
    "why_sync": "TLP spans all 4 lanes byte-striped; 8 UI max deskew buffer; "
                "beyond that the receiver cannot reconstruct the original byte order",
}


# ===================================================================
# 7. Music & Media
# ===================================================================

ORCHESTRAL_PERFORMANCE = {
    "name": "Boccherini Minuet — String Quintet (actual notes)",
    "description": "Exact note-level session types extracted from the MIDI score "
                   "of Boccherini's Minuet from String Quintet in E major, Op. 11 No. 5. "
                   "Each voice is the actual pitch sequence from the score.",
    "voices": [
        # Violin I (melody) — A5 Gs5 A5 B5 A5 A4 Cs5 E5 E5 D5 D5 D5
        "&{A5: &{Gs5: &{A5: &{B5: &{A5: &{A4: &{Cs5: &{E5: &{E5: &{D5: &{D5: &{D5: end}}}}}}}}}}}}",
        # Violin II (ostinato) — E5-E4 repeating
        "rec X . &{E5: &{E4: X}, done: end}",
        # Viola — A4 Cs5 A4 E4 A4 Cs5 Gs4 B4 Gs4 B4 Gs4 B4
        "&{A4: &{Cs5: &{A4: &{E4: &{A4: &{Cs5: &{Gs4: &{B4: &{Gs4: &{B4: &{Gs4: &{B4: end}}}}}}}}}}}}",
        # Violoncello I (bass ostinato) — Cs4-A3 repeating
        "rec X . &{Cs4: &{A3: X}, done: end}",
        # Violoncello II (bass line) — A2 A2 A2 E3 E3 E3 E2 E2 E2 A2 A2 A2
        "&{A2: &{A2: &{A2: &{E3: &{E3: &{E3: &{E2: &{E2: &{E2: &{A2: &{A2: &{A2: end}}}}}}}}}}}}",
    ],
    "why_sync": "All five voices advance beat-by-beat together; the Violin II "
                "E5-E4 ostinato must align with Cello I Cs4-A3 bass on every beat "
                "or the harmonic structure collapses",
}

AV_SYNC = {
    "name": "HDMI 2.1 Audio-Video Transport",
    "description": "48 Gbps video + eARC audio on HDMI 2.1 link. "
                   "VSYNC-locked audio embedding per CTA-861-H.",
    "voices": [
        # Video pipeline
        "rec X . &{vsyncPulse: &{activeRegion: &{pixelStream: &{blankingInterval: &{infoFrame: X}}}}, hotplugDetect: end}",
        # Audio pipeline (embedded in blanking)
        "rec X . &{vsyncPulse: &{audioSample: &{audioPacket: &{insertInBlanking: &{channelStatus: X}}}}, hotplugDetect: end}",
    ],
    "why_sync": "Audio is embedded in video blanking interval; audio PLL is locked "
                "to video pixel clock via N/CTS; async drift >2ms is audible",
}

MULTICAM_BROADCAST = {
    "name": "Multi-Camera OB Van (SMPTE ST 2110)",
    "description": "Three broadcast cameras with PTP (IEEE 1588) genlock. "
                   "Based on SMPTE ST 2110-10 RTP essence transport.",
    "voices": [
        # Camera 1 (4K HDR)
        "rec X . &{ptpSync: &{sensorExpose: &{debayer: &{colorGrade: &{rtpPacketize: &{transmit: X}}}}}, standby: end}",
        # Camera 2 (4K HDR)
        "rec X . &{ptpSync: &{sensorExpose: &{debayer: &{colorGrade: &{rtpPacketize: &{transmit: X}}}}}, standby: end}",
        # Camera 3 (4K HDR)
        "rec X . &{ptpSync: &{sensorExpose: &{debayer: &{colorGrade: &{rtpPacketize: &{transmit: X}}}}}, standby: end}",
    ],
    "why_sync": "Vision mixer cut requires frame-aligned inputs; PTP accuracy <1μs; "
                "frame boundary misalignment → torn frame on cut → broadcast violation",
}


# ===================================================================
# 8. Financial
# ===================================================================

ATOMIC_CROSS_LEDGER = {
    "name": "SWIFT gpi Cross-Border Transfer (ISO 20022)",
    "description": "Correspondent banking: debit nostro at Bank A, credit vostro "
                   "at Bank B. Based on SWIFT gpi with uetr tracking.",
    "voices": [
        # Originating bank (nostro debit)
        "&{pacs008Create: &{sanctionsScreen: +{screenPass: &{nostroDebit: &{gpiConfirm: +{uetrMatch: &{creditConfirm: end}, uetrTimeout: &{investigate: end}}}}, screenHit: &{holdFunds: end}}}}",
        # Beneficiary bank (vostro credit)
        "&{pacs008Receive: &{sanctionsScreen: +{screenPass: &{vostroCredit: &{gpiConfirm: +{uetrMatch: &{creditAdvice: end}, uetrTimeout: &{returnFunds: end}}}}, screenHit: &{rejectPayment: end}}}}",
    ],
    "why_sync": "Nostro debit without vostro credit = money in transit with no owner; "
                "regulatory requirement: debit-credit must settle within same value date",
}

MARKET_CLOSING_AUCTION = {
    "name": "Exchange Closing Auction (Euronext UTP)",
    "description": "Three order books frozen simultaneously for indicative "
                   "price calculation. Based on Euronext UTP protocol.",
    "voices": [
        # Order book A (large-cap equity)
        "&{tradingPhase: &{auctionCall: &{freezeBook: &{indicativeCalc: &{uncrossing: +{filled: &{confirmTrades: end}, noMatch: &{expireOrders: end}}}}}}}",
        # Order book B (mid-cap equity)
        "&{tradingPhase: &{auctionCall: &{freezeBook: &{indicativeCalc: &{uncrossing: +{filled: &{confirmTrades: end}, noMatch: &{expireOrders: end}}}}}}}",
        # Order book C (ETF)
        "&{tradingPhase: &{auctionCall: &{freezeBook: &{indicativeCalc: &{uncrossing: +{filled: &{confirmTrades: end}, noMatch: &{expireOrders: end}}}}}}}",
    ],
    "why_sync": "Staggered freeze enables cross-book arbitrage during the gap; "
                "MiFID II requires simultaneous auction across related instruments",
}


# ===================================================================
# Collection
# ===================================================================

ALL_SYNC_BENCHMARKS = [
    AUTOMOTIVE_WELDING,
    BOTTLING_PLANT,
    SEMICONDUCTOR_LITHO,
    TRIPLE_MODULAR_REDUNDANCY,
    FLY_BY_WIRE,
    ABS_BRAKING,
    TDM_MULTIPLEXING,
    BEAMFORMING_5G,
    DUAL_ARM_MANIPULATION,
    HUMANOID_WALKING,
    DRONE_SWARM_FORMATION,
    TWO_PHASE_COMMIT,
    RAFT_LOG_REPLICATION,
    MPI_BARRIER,
    SYNCHRONOUS_CIRCUIT,
    DDR_MEMORY,
    PCIE_LANES,
    ORCHESTRAL_PERFORMANCE,
    AV_SYNC,
    MULTICAM_BROADCAST,
    ATOMIC_CROSS_LEDGER,
    MARKET_CLOSING_AUCTION,
]
