"""Session type registry for ALL BICA Reborn classes (Step 80f).

Every class in the BICA Reborn Java codebase has a session type:
  - MANUAL: hand-written from class documentation
  - EXTRACTED: generated from method signatures + return types
  - DERIVED: inferred from usage patterns in the codebase

Three categories:
  - STATEFUL: classes with lifecycle (open/use/close pattern) → full session type
  - CHECKER: analysis classes (construct → analyze → get result) → pipeline session type
  - DATA: immutable records, exceptions, value objects → trivial session type (end)

This is the complete session type annotation of a real Java project.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ClassSessionType:
    """Session type annotation for a BICA Reborn class."""
    class_name: str
    package: str
    category: str  # "stateful", "checker", "data", "builder", "agent"
    session_type: str
    origin: str  # "manual", "extracted", "derived"
    description: str


# ---------------------------------------------------------------------------
# Complete session type registry for BICA Reborn (58 classes)
# ---------------------------------------------------------------------------

BICA_SESSION_TYPES: list[ClassSessionType] = [

    # ===== SAMPLE CLASSES (stateful objects with clear protocols) =====

    ClassSessionType(
        class_name="FileHandle",
        package="com.bica.reborn.samples",
        category="stateful",
        session_type="&{open: rec X . &{read: X, write: X, close: end}}",
        origin="manual",
        description="File handle: open → read/write loop → close",
    ),
    ClassSessionType(
        class_name="SimpleIterator",
        package="com.bica.reborn.samples",
        category="stateful",
        session_type="rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}",
        origin="manual",
        description="Iterator: hasNext selects TRUE/FALSE, next on TRUE",
    ),
    ClassSessionType(
        class_name="ATM",
        package="com.bica.reborn.samples",
        category="stateful",
        session_type="&{insertCard: &{enterPIN: +{AUTH: rec X . &{checkBalance: X, withdraw: +{OK: X, INSUFFICIENT: X}, deposit: X, ejectCard: end}, DENIED: end}}}",
        origin="manual",
        description="ATM: insert → PIN → auth select → operations loop → eject",
    ),

    # ===== PIPELINE CLASSES (parse → analyze → get result) =====

    ClassSessionType(
        class_name="AnalysisPipeline",
        package="com.bica.reborn.samples",
        category="checker",
        session_type="&{parse: &{buildStateSpace: &{checkLattice: &{checkDistributive: &{getResult: end}}}}}",
        origin="manual",
        description="Analysis pipeline: parse → build → check lattice → check dist → result",
    ),
    ClassSessionType(
        class_name="ParserSession",
        package="com.bica.reborn.samples",
        category="checker",
        session_type="&{setInput: &{parse: +{OK: &{prettyPrint: end, getAST: end}, ERROR: &{getError: end}}}}",
        origin="manual",
        description="Parser session: set input → parse → success/error branches",
    ),
    ClassSessionType(
        class_name="SubtypingSession",
        package="com.bica.reborn.samples",
        category="checker",
        session_type="&{setLeft: &{setRight: &{checkSubtype: end}}}",
        origin="manual",
        description="Subtyping check: set left → set right → check",
    ),
    ClassSessionType(
        class_name="MorphismSession",
        package="com.bica.reborn.samples",
        category="checker",
        session_type="&{setSource: &{setTarget: &{classify: end}}}",
        origin="manual",
        description="Morphism classification: set source → set target → classify",
    ),
    ClassSessionType(
        class_name="DualitySession",
        package="com.bica.reborn.samples",
        category="checker",
        session_type="&{setType: &{computeDual: &{checkInvolution: end, getDual: end}}}",
        origin="manual",
        description="Duality session: set type → compute dual → check/get",
    ),
    ClassSessionType(
        class_name="RecursionAnalysisSession",
        package="com.bica.reborn.samples",
        category="checker",
        session_type="&{setType: &{checkGuardedness: &{checkContractivity: &{analyze: end}}}}",
        origin="manual",
        description="Recursion analysis: set type → guarded → contractive → analyze",
    ),
    ClassSessionType(
        class_name="CoverageSession",
        package="com.bica.reborn.samples",
        category="checker",
        session_type="&{parseType: &{enumeratePaths: &{computeCoverage: &{getStateCoverage: end, getTransitionCoverage: end}}}}",
        origin="manual",
        description="Coverage session: parse → enumerate → compute → get metrics",
    ),
    ClassSessionType(
        class_name="EndomorphismSession",
        package="com.bica.reborn.samples",
        category="checker",
        session_type="&{parseType: &{analyzeEndomorphisms: &{getSummary: end, getPreservationRate: end}}}",
        origin="manual",
        description="Endomorphism session: parse → analyze → get results",
    ),
    ClassSessionType(
        class_name="TestGenSession",
        package="com.bica.reborn.samples",
        category="checker",
        session_type="&{parseType: &{enumeratePaths: &{getPathCount: end, generateTests: end}}}",
        origin="manual",
        description="Test generation session: parse → enumerate → count/generate",
    ),
    ClassSessionType(
        class_name="EnumerationSession",
        package="com.bica.reborn.samples",
        category="checker",
        session_type="&{configure: &{enumerate: +{UNIVERSAL: &{getCount: end}, COUNTEREXAMPLE: &{getCounterexample: end}}}}",
        origin="manual",
        description="Enumeration session: configure → enumerate → universal/counterexample",
    ),
    ClassSessionType(
        class_name="GlobalTypeSession",
        package="com.bica.reborn.samples",
        category="checker",
        session_type="&{parseGlobal: &{buildGlobalStateSpace: &{projectAll: &{checkProjections: end, checkLattice: end}}}}",
        origin="manual",
        description="Global type session: parse → build → project → check",
    ),

    # ===== CORE CHECKER CLASSES (stateless analysis — take StateSpace, return result) =====

    ClassSessionType(
        class_name="LatticeChecker",
        package="com.bica.reborn.lattice",
        category="checker",
        session_type="&{checkLattice: end, buildQuotientPoset: end}",
        origin="extracted",
        description="Lattice checker: single-shot analysis methods",
    ),
    ClassSessionType(
        class_name="IrreduciblesChecker",
        package="com.bica.reborn.lattice",
        category="checker",
        session_type="&{lowerCovers: end, upperCovers: end, joinIrreducibles: end, meetIrreducibles: end, isJoinIrreducible: end}",
        origin="extracted",
        description="Irreducibles checker: query methods on lattice structure",
    ),
    ClassSessionType(
        class_name="ReticularChecker",
        package="com.bica.reborn.reticular",
        category="checker",
        session_type="&{classifyState: end, classifyAllStates: end, reconstruct: end}",
        origin="extracted",
        description="Reticular form checker",
    ),
    ClassSessionType(
        class_name="Reconstructor",
        package="com.bica.reborn.reticular",
        category="checker",
        session_type="&{reconstruct: end}",
        origin="extracted",
        description="Session type reconstructor from state space",
    ),
    ClassSessionType(
        class_name="BirkhoffChecker",
        package="com.bica.reborn.birkhoff",
        category="checker",
        session_type="&{birkhoffRepresentation: end}",
        origin="extracted",
        description="Birkhoff representation checker",
    ),
    ClassSessionType(
        class_name="AlgebraicChecker",
        package="com.bica.reborn.algebraic",
        category="checker",
        session_type="&{stateList: end, reachability: end, computeMobiusMatrix: end, computeMobiusValue: end, computeRotaPolynomial: end}",
        origin="extracted",
        description="Algebraic invariants: query-phase, all methods always available",
    ),
    ClassSessionType(
        class_name="CSPChecker",
        package="com.bica.reborn.csp",
        category="checker",
        session_type="&{extractTraces: end, extractCompleteTraces: end, alphabet: end, failures: end, isDeterministic: end, computeRefusals: end}",
        origin="extracted",
        description="CSP semantics checker: all query methods in any order",
    ),
    ClassSessionType(
        class_name="FailureSemantics",
        package="com.bica.reborn.csp",
        category="checker",
        session_type="&{analyzeFailures: end}",
        origin="extracted",
        description="Failure semantics analysis",
    ),
    ClassSessionType(
        class_name="CategoryChecker",
        package="com.bica.reborn.category",
        category="checker",
        session_type="&{identityMorphism: end, isProduct: end, checkProductStructure: end}",
        origin="extracted",
        description="Category theory checker: query methods",
    ),
    ClassSessionType(
        class_name="EndomorphismChecker",
        package="com.bica.reborn.endomorphism",
        category="checker",
        session_type="&{extractTransitionMaps: &{checkEndomorphism: end}}",
        origin="extracted",
        description="Endomorphism checker: extract maps → check properties",
    ),
    ClassSessionType(
        class_name="PolarityChecker",
        package="com.bica.reborn.polarity",
        category="checker",
        session_type="&{buildIncidenceRelation: &{allLabels: end}}",
        origin="extracted",
        description="Polarity checker: build relation → get labels",
    ),
    ClassSessionType(
        class_name="ProcessEquivalenceChecker",
        package="com.bica.reborn.equivalence",
        category="checker",
        session_type="&{subStateSpace: end, stateSpaceToLTS: end}",
        origin="extracted",
        description="Process equivalence: extraction utilities",
    ),
    ClassSessionType(
        class_name="CoequalizerChecker",
        package="com.bica.reborn.coequalizer",
        category="checker",
        session_type="&{find: end, union: end, classes: end}",
        origin="extracted",
        description="Coequalizer via union-find",
    ),
    ClassSessionType(
        class_name="PetriNetBuilder",
        package="com.bica.reborn.petri",
        category="checker",
        session_type="&{buildPetriNet: &{verifyIsomorphism: end, checkAcyclic: end}}",
        origin="extracted",
        description="Petri net builder: build → verify properties",
    ),
    ClassSessionType(
        class_name="CoverabilityChecker",
        package="com.bica.reborn.petri",
        category="checker",
        session_type="end",
        origin="extracted",
        description="Coverability checker: stateless record-based",
    ),
    ClassSessionType(
        class_name="MarkingLatticeChecker",
        package="com.bica.reborn.marking",
        category="checker",
        session_type="&{computeHeight: end, countMaximalChains: end, computeCoveringRelation: end, computeReachability: end, checkMarkingIsomorphism: end, computeWidthApprox: end}",
        origin="extracted",
        description="Marking lattice analysis: all methods available in any order",
    ),
    ClassSessionType(
        class_name="HasseDiagram",
        package="com.bica.reborn.visualize",
        category="checker",
        session_type="&{dotSource: end, hasseEdges: end}",
        origin="extracted",
        description="Hasse diagram generator",
    ),
    ClassSessionType(
        class_name="TensorChecker",
        package="com.bica.reborn.tensor",
        category="checker",
        session_type="&{buildTensorNary: &{buildProjections: end}}",
        origin="extracted",
        description="Tensor product checker: build → project",
    ),

    # ===== MONITOR (stateful — tracks session progress) =====

    ClassSessionType(
        class_name="SessionMonitor",
        package="com.bica.reborn.monitor",
        category="stateful",
        session_type="rec X . &{step: +{OK: X, VIOLATION: X, COMPLETE: end}, isComplete: X, isViolation: X, currentState: X, enabledMethods: X, enabledSelections: X, reset: X, trace: X, buildTransitionMap: X}",
        origin="manual",
        description="Session monitor: step through protocol, check status, reset",
    ),

    # ===== AGENT CLASSES (handle messages, dispatch) =====

    ClassSessionType(
        class_name="SimpleAgent",
        package="com.bica.reborn.agent",
        category="agent",
        session_type="&{protocol: &{name: &{handle: end}}}",
        origin="extracted",
        description="Simple agent: get protocol → get name → handle message",
    ),
    ClassSessionType(
        class_name="BranchHandler",
        package="com.bica.reborn.agent.handler",
        category="agent",
        session_type="&{handle: end}",
        origin="extracted",
        description="Branch handler: dispatch to registered handler",
    ),
    ClassSessionType(
        class_name="Builder",
        package="com.bica.reborn.agent.handler",
        category="builder",
        session_type="rec X . &{on: X, fallback: X, build: end}",
        origin="manual",
        description="Handler builder: register handlers → build",
    ),
    ClassSessionType(
        class_name="A2aTransport",
        package="com.bica.reborn.agent.transport",
        category="agent",
        session_type="&{send: +{OK: end, ERR: end}, onMessage: end, registerHandler: end, getTask: +{FOUND: end, EMPTY: end}, allTasks: end, id: end, listenerCount: end}",
        origin="manual",
        description="A2A transport: send/receive messages, manage tasks",
    ),
    ClassSessionType(
        class_name="InProcessTransport",
        package="com.bica.reborn.agent.transport",
        category="agent",
        session_type="&{send: +{OK: end, ERR: end}, onMessage: end, id: end, listenerCount: end}",
        origin="manual",
        description="In-process transport: local message passing",
    ),

    # ===== WORKFLOW =====

    ClassSessionType(
        class_name="Orchestrator",
        package="com.bica.reborn.workflow",
        category="stateful",
        session_type="rec X . &{registerHandler: X, orchestrate: &{collectResults: end}, dispatchTask: X, sprintCount: X}",
        origin="manual",
        description="Workflow orchestrator: register → orchestrate → collect",
    ),

    # ===== PROTOCOL MINER =====

    ClassSessionType(
        class_name="ProtocolMiner",
        package="com.bica.reborn.domain.mining",
        category="checker",
        session_type="&{mineFromTraces: end, mineFromStateSpace: end}",
        origin="extracted",
        description="Protocol mining: single-shot analysis methods",
    ),

    # ===== UNION-FIND (utility) =====

    ClassSessionType(
        class_name="UnionFind",
        package="com.bica.reborn.coequalizer",
        category="data",
        session_type="&{find: end, union: end, classes: end}",
        origin="extracted",
        description="Union-find: query/modify in any order",
    ),

    # ===== PARSER =====

    ClassSessionType(
        class_name="ParseError",
        package="com.bica.reborn.parser",
        category="data",
        session_type="&{getPos: end}",
        origin="extracted",
        description="Parse error: get position (immutable)",
    ),
    ClassSessionType(
        class_name="Composer",
        package="com.bica.reborn.composition",
        category="checker",
        session_type="&{composeFromSections: end}",
        origin="extracted",
        description="Composition: single-shot compose method",
    ),

    # ===== EXCEPTION / DATA CLASSES (trivial session types) =====

    ClassSessionType(
        class_name="ProtocolViolationException",
        package="com.bica.reborn.exception",
        category="data",
        session_type="&{attemptedMethod: end, currentState: end, enabledMethods: end}",
        origin="extracted",
        description="Exception: query fields (immutable)",
    ),
    ClassSessionType(
        class_name="IncompleteSessionException",
        package="com.bica.reborn.exception",
        category="data",
        session_type="&{currentState: end, remainingMethods: end}",
        origin="extracted",
        description="Exception: query fields (immutable)",
    ),
    ClassSessionType(
        class_name="ProjectionError",
        package="com.bica.reborn.globaltype",
        category="data",
        session_type="&{role: end}",
        origin="extracted",
        description="Projection error: get role (immutable)",
    ),

    # ===== REMAINING 12 CLASSES =====

    ClassSessionType(
        class_name="AgentCatalog",
        package="com.bica.reborn.agent",
        category="agent",
        session_type="&{register: end, lookup: +{FOUND: end, NOT_FOUND: end}, list: end}",
        origin="derived",
        description="Agent catalog: register/lookup/list agents",
    ),
    ClassSessionType(
        class_name="AgentException",
        package="com.bica.reborn.agent",
        category="data",
        session_type="end",
        origin="derived",
        description="Agent exception: immutable error",
    ),
    ClassSessionType(
        class_name="AgentVerifier",
        package="com.bica.reborn.agent",
        category="checker",
        session_type="&{verify: +{OK: end, VIOLATION: end}}",
        origin="derived",
        description="Agent verifier: verify → OK/VIOLATION",
    ),
    ClassSessionType(
        class_name="AudioRoutingChecker",
        package="com.bica.reborn.audio",
        category="checker",
        session_type="&{checkRouting: end, buildRoutingGraph: end}",
        origin="derived",
        description="Audio routing checker: analyze routing graph",
    ),
    ClassSessionType(
        class_name="ChannelChecker",
        package="com.bica.reborn.channel",
        category="checker",
        session_type="&{checkChannel: end, checkBuffered: end, checkAsync: end}",
        origin="derived",
        description="Channel checker: verify channel properties",
    ),
    ClassSessionType(
        class_name="ConcurrencyLattice",
        package="com.bica.reborn.concurrency",
        category="checker",
        session_type="&{buildProduct: end, checkIndependence: end}",
        origin="derived",
        description="Concurrency lattice: product construction + independence check",
    ),
    ClassSessionType(
        class_name="CoverageChecker",
        package="com.bica.reborn.coverage",
        category="checker",
        session_type="&{computeCoverage: end}",
        origin="derived",
        description="Coverage checker: compute test coverage",
    ),
    ClassSessionType(
        class_name="EqualizerChecker",
        package="com.bica.reborn.equalizer",
        category="checker",
        session_type="&{findEqualizer: end}",
        origin="derived",
        description="Equalizer checker: find equalizer in category",
    ),
    ClassSessionType(
        class_name="MonoidalChecker",
        package="com.bica.reborn.monoidal",
        category="checker",
        session_type="&{checkMonoidal: end, tensorUnit: end}",
        origin="derived",
        description="Monoidal category checker",
    ),
    ClassSessionType(
        class_name="ProductStateSpace",
        package="com.bica.reborn.product",
        category="checker",
        session_type="&{buildProduct: end, getLeft: end, getRight: end}",
        origin="derived",
        description="Product state space: build and project",
    ),
    ClassSessionType(
        class_name="ReplicationChecker",
        package="com.bica.reborn.replication",
        category="checker",
        session_type="&{checkReplication: end, unfoldReplication: end}",
        origin="derived",
        description="Replication checker: !S modality",
    ),
    ClassSessionType(
        class_name="TrieNode",
        package="com.bica.reborn.domain.mining",
        category="data",
        session_type="&{getChildren: end, isTerminal: end}",
        origin="derived",
        description="Trie node: internal data structure for mining",
    ),
]


def get_session_type(class_name: str) -> Optional[ClassSessionType]:
    """Look up the session type for a BICA class by name."""
    for st in BICA_SESSION_TYPES:
        if st.class_name == class_name:
            return st
    return None


def summary() -> str:
    """Print summary of session type coverage."""
    total = len(BICA_SESSION_TYPES)
    by_category = {}
    by_origin = {}
    for st in BICA_SESSION_TYPES:
        by_category[st.category] = by_category.get(st.category, 0) + 1
        by_origin[st.origin] = by_origin.get(st.origin, 0) + 1

    lines = [
        f"BICA Reborn Session Type Registry: {total} classes",
        "",
        "By category:",
    ]
    for cat, count in sorted(by_category.items()):
        lines.append(f"  {cat}: {count}")
    lines.append("")
    lines.append("By origin:")
    for orig, count in sorted(by_origin.items()):
        lines.append(f"  {orig}: {count}")
    return "\n".join(lines)
