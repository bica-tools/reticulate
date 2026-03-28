"""Encyclopedia of Civilization as Session Types (Step 212).

Every institution, every societal process, every collective human endeavor
follows a protocol.  Elections have regulated phases.  Trials proceed
through structured stages.  Markets execute transactions via standardized
exchanges.  Education transmits knowledge through ritualized interactions.

The key insight: civilization IS a collection of session types running
concurrently.  The protocols that govern governance, economics, education,
medicine, technology, and justice are not merely analogous to session types
-- they ARE session types, evolved over millennia to coordinate human
behavior at scale.

This module provides:
    ``CIVILIZATION_ENCYCLOPEDIA``      -- dict of name -> CivilizationEntry.
    ``civilization_by_domain(domain)`` -- find entries in a domain.
    ``all_civilization_form_lattices()`` -- verify every entry forms a lattice.
"""

from __future__ import annotations

from dataclasses import dataclass

from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.lattice import check_lattice


# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CivilizationEntry:
    """A single entry in the civilization encyclopedia.

    Attributes:
        name: Unique identifier for this civilizational process.
        domain: Societal domain (governance, economics, education, medicine,
                technology, justice).
        session_type_str: Session type in textual syntax.
        description: Short human-readable description.
    """

    name: str
    domain: str
    session_type_str: str
    description: str


# ---------------------------------------------------------------------------
# The Civilization Encyclopedia
# ---------------------------------------------------------------------------

CIVILIZATION_ENCYCLOPEDIA: dict[str, CivilizationEntry] = {
    # ======================================================================
    # GOVERNANCE (10 entries)
    # ======================================================================
    "election": CivilizationEntry(
        "election", "governance",
        "&{announce: +{nominate: &{campaign: +{vote: &{count: +{certify: &{inaugurate: end}}}}}}}",
        "Democratic election from announcement through campaigning to inauguration.",
    ),
    "legislation": CivilizationEntry(
        "legislation", "governance",
        "&{draft_bill: +{committee_review: &{floor_debate: +{vote: &{sign_into_law: end, veto: end}}}}}",
        "Legislative process from bill drafting through debate to enactment or veto.",
    ),
    "court_trial": CivilizationEntry(
        "court_trial", "governance",
        "&{arraign: +{plead: &{present_evidence: +{cross_examine: &{deliberate: +{verdict: &{sentence: end}}}}}}}",
        "Court trial from arraignment through evidence to verdict and sentencing.",
    ),
    "treaty_negotiation": CivilizationEntry(
        "treaty_negotiation", "governance",
        "&{propose: +{negotiate: &{draft: +{review: &{sign: +{ratify: end, reject: end}}}}}}",
        "International treaty from proposal through negotiation to ratification.",
    ),
    "revolution": CivilizationEntry(
        "revolution", "governance",
        "&{grievance: +{organize: &{protest: +{escalate: &{overthrow: +{establish_new: end}}}}}}",
        "Revolutionary process from grievance through organization to regime change.",
    ),
    "coup": CivilizationEntry(
        "coup", "governance",
        "&{conspire: +{mobilize: &{seize_power: +{secure: &{announce: end}}}}}",
        "Coup: conspiracy, mobilization, power seizure, and announcement.",
    ),
    "referendum": CivilizationEntry(
        "referendum", "governance",
        "&{propose_question: +{campaign: &{vote: +{count: &{implement: end, reject_status_quo: end}}}}}",
        "Direct democratic vote on a specific question.",
    ),
    "impeachment": CivilizationEntry(
        "impeachment", "governance",
        "&{investigate: +{charge: &{house_vote: +{senate_trial: &{convict: end, acquit: end}}}}}",
        "Impeachment process from investigation through trial to conviction or acquittal.",
    ),
    "constitutional_amendment": CivilizationEntry(
        "constitutional_amendment", "governance",
        "&{propose: +{two_thirds_vote: &{state_ratification: +{certify: end}}}}",
        "Constitutional amendment requiring supermajority and state ratification.",
    ),
    "census": CivilizationEntry(
        "census", "governance",
        "&{design_survey: +{distribute: &{collect: +{tabulate: &{publish: end}}}}}",
        "Population census: design, distribute, collect, tabulate, publish.",
    ),

    # ======================================================================
    # ECONOMICS (10 entries)
    # ======================================================================
    "market_transaction": CivilizationEntry(
        "market_transaction", "economics",
        "&{list_product: +{buyer_browse: &{negotiate_price: +{agree: &{pay: +{deliver: end}}}}}}",
        "Market transaction from listing through price negotiation to delivery.",
    ),
    "auction": CivilizationEntry(
        "auction", "economics",
        "rec X . &{present_lot: +{open_bidding: &{bid: +{raise: X, hammer: end}}}}",
        "Auction as recursive bidding until the hammer falls.",
    ),
    "stock_trade": CivilizationEntry(
        "stock_trade", "economics",
        "&{place_order: +{match: &{execute: +{clear: &{settle: end}}}}}",
        "Stock trade: order placement, matching, execution, clearing, settlement.",
    ),
    "loan": CivilizationEntry(
        "loan", "economics",
        "&{apply: +{assess_credit: &{approve: +{disburse: &{repay: +{close_account: end}}}, deny: end}}}",
        "Loan lifecycle from application through assessment to repayment or denial.",
    ),
    "insurance_claim": CivilizationEntry(
        "insurance_claim", "economics",
        "&{incident: +{file_claim: &{investigate: +{approve: &{pay_out: end}, deny_claim: end}}}}",
        "Insurance claim from incident through investigation to payout or denial.",
    ),
    "bankruptcy": CivilizationEntry(
        "bankruptcy", "economics",
        "&{insolvency: +{file_petition: &{list_assets: +{liquidate: &{distribute: +{discharge: end}}}}}}",
        "Bankruptcy from insolvency through petition to asset liquidation and discharge.",
    ),
    "startup_founding": CivilizationEntry(
        "startup_founding", "economics",
        "&{idea: +{validate: &{incorporate: +{fund_raise: &{build_product: +{launch: end}}}}}}",
        "Startup founding from idea validation through funding to product launch.",
    ),
    "ipo": CivilizationEntry(
        "ipo", "economics",
        "&{prepare_financials: +{select_underwriter: &{roadshow: +{price: &{list_exchange: +{trade_begins: end}}}}}}",
        "Initial public offering from financial preparation to exchange listing.",
    ),
    "merger_acquisition": CivilizationEntry(
        "merger_acquisition", "economics",
        "&{identify_target: +{due_diligence: &{negotiate_terms: +{regulatory_approval: &{close_deal: +{integrate: end}}}}}}",
        "M&A from target identification through due diligence to integration.",
    ),
    "supply_chain": CivilizationEntry(
        "supply_chain", "economics",
        "rec X . &{order_raw: +{manufacture: &{ship: +{warehouse: &{distribute: +{sell: X, discontinue: end}}}}}}",
        "Supply chain as recursive cycle from raw materials to retail.",
    ),

    # ======================================================================
    # EDUCATION (8 entries)
    # ======================================================================
    "kindergarten_day": CivilizationEntry(
        "kindergarten_day", "education",
        "&{arrival: +{circle_time: &{activity: +{snack: &{play: +{story: &{dismissal: end}}}}}}}",
        "Kindergarten day: circle time, activity, snack, play, story, dismissal.",
    ),
    "school_lesson": CivilizationEntry(
        "school_lesson", "education",
        "&{bell: +{introduce_topic: &{explain: +{practice: &{assess: +{dismiss: end}}}}}}",
        "Classroom lesson: introduction, explanation, practice, assessment.",
    ),
    "university_lecture": CivilizationEntry(
        "university_lecture", "education",
        "&{enter_hall: +{lecture: &{examples: +{questions: &{summary: end}}}}}",
        "University lecture: delivery, examples, Q&A, summary.",
    ),
    "thesis_defense": CivilizationEntry(
        "thesis_defense", "education",
        "&{present: +{committee_questions: &{respond: +{deliberate: &{pass: end, revise: end}}}}}",
        "Thesis defense from presentation through questioning to pass or revise.",
    ),
    "apprenticeship": CivilizationEntry(
        "apprenticeship", "education",
        "rec X . &{observe: +{attempt: &{feedback: +{improve: X, master: end}}}}",
        "Apprenticeship as recursive cycle of observation, attempt, and feedback.",
    ),
    "online_course": CivilizationEntry(
        "online_course", "education",
        "rec X . &{watch_lecture: +{take_quiz: &{pass: +{next_module: X, complete: end}, fail: +{review: X}}}}",
        "Online course: recursive module completion with quiz gates.",
    ),
    "exam": CivilizationEntry(
        "exam", "education",
        "&{enter_room: +{read_questions: &{answer: +{submit: &{grade: +{return_results: end}}}}}}",
        "Examination from entry through answering to grading and results.",
    ),
    "peer_review": CivilizationEntry(
        "peer_review", "education",
        "&{submit: +{assign_reviewers: &{review: +{feedback: &{revise: +{accept: end, reject: end}}}}}}",
        "Peer review from submission through review to acceptance or rejection.",
    ),

    # ======================================================================
    # MEDICINE (8 entries)
    # ======================================================================
    "doctor_visit": CivilizationEntry(
        "doctor_visit", "medicine",
        "&{check_in: +{wait: &{examine: +{diagnose: &{prescribe: +{follow_up: end}}}}}}",
        "Doctor visit: check-in, examination, diagnosis, prescription.",
    ),
    "surgery": CivilizationEntry(
        "surgery", "medicine",
        "&{prep_patient: +{anesthesia: &{incision: +{procedure: &{close: +{recovery: end}}}}}}",
        "Surgery from preparation through anesthesia and procedure to recovery.",
    ),
    "vaccination": CivilizationEntry(
        "vaccination", "medicine",
        "&{register: +{consent: &{inject: +{observe: &{immune_response: end}}}}}",
        "Vaccination: registration, consent, injection, observation, immune response.",
    ),
    "rehabilitation": CivilizationEntry(
        "rehabilitation", "medicine",
        "rec X . &{assess: +{therapy: &{exercise: +{progress: X, discharge: end}}}}",
        "Rehabilitation as recursive assessment and therapy until discharge.",
    ),
    "clinical_trial": CivilizationEntry(
        "clinical_trial", "medicine",
        "&{design: +{recruit: &{randomize: +{treat: &{measure: +{analyze: &{publish: end}}}}}}}",
        "Clinical trial from design through randomized treatment to publication.",
    ),
    "organ_transplant": CivilizationEntry(
        "organ_transplant", "medicine",
        "&{match_donor: +{harvest: &{transport: +{implant: &{monitor_rejection: +{stable: end}}}}}}",
        "Organ transplant from donor matching through implantation to stability.",
    ),
    "childbirth": CivilizationEntry(
        "childbirth", "medicine",
        "&{admit: +{labor: &{deliver: +{cord_cut: &{apgar: +{bond: end}}}}}}",
        "Medical childbirth from admission through delivery to bonding.",
    ),
    "emergency_room": CivilizationEntry(
        "emergency_room", "medicine",
        "&{triage: +{assess: &{stabilize: +{treat: &{admit: end, discharge: end}}}}}",
        "Emergency care: triage, assessment, stabilization, treatment, disposition.",
    ),

    # ======================================================================
    # TECHNOLOGY (8 entries)
    # ======================================================================
    "software_development": CivilizationEntry(
        "software_development", "technology",
        "rec X . &{requirements: +{design: &{implement: +{test: &{deploy: +{maintain: X, sunset: end}}}}}}",
        "Software development lifecycle: recursive build-deploy-maintain.",
    ),
    "factory_production": CivilizationEntry(
        "factory_production", "technology",
        "rec X . &{source_material: +{fabricate: &{assemble: +{quality_check: &{ship: X}, reject: end}}}}",
        "Factory production as recursive sourcing, fabrication, and shipping.",
    ),
    "space_launch": CivilizationEntry(
        "space_launch", "technology",
        "&{countdown: +{ignition: &{liftoff: +{stage_separate: &{orbit_insert: +{mission: end}}}}}}",
        "Space launch from countdown through staging to orbit insertion.",
    ),
    "internet_search": CivilizationEntry(
        "internet_search", "technology",
        "&{query: +{index_lookup: &{rank_results: +{display: &{click: end}}}}}",
        "Internet search: query, index lookup, ranking, display, click.",
    ),
    "social_media_post": CivilizationEntry(
        "social_media_post", "technology",
        "&{compose: +{publish: &{distribute: +{engage: &{archive: end}}}}}",
        "Social media post: compose, publish, distribute, engage, archive.",
    ),
    "video_call": CivilizationEntry(
        "video_call", "technology",
        "&{initiate: +{connect: &{stream: +{converse: &{disconnect: end}}}}}",
        "Video call: initiate, connect, stream, converse, disconnect.",
    ),
    "autonomous_driving": CivilizationEntry(
        "autonomous_driving", "technology",
        "rec X . &{sense: +{perceive: &{plan: +{actuate: X, park: end}}}}",
        "Autonomous driving as recursive sense-perceive-plan-actuate loop.",
    ),
    "three_d_printing": CivilizationEntry(
        "three_d_printing", "technology",
        "&{design_model: +{slice_layers: &{print_layer: +{cure: &{finish: end}}}}}",
        "3D printing: model design, slicing, layer printing, curing, finishing.",
    ),

    # ======================================================================
    # JUSTICE (6 entries)
    # ======================================================================
    "arrest": CivilizationEntry(
        "arrest", "justice",
        "&{probable_cause: +{detain: &{read_rights: +{transport: &{book: end}}}}}",
        "Arrest procedure: probable cause, detention, Miranda rights, booking.",
    ),
    "investigation": CivilizationEntry(
        "investigation", "justice",
        "rec X . &{gather_evidence: +{analyze: &{interview: +{lead: X, conclude: end}}}}",
        "Criminal investigation as recursive evidence gathering until conclusion.",
    ),
    "prosecution": CivilizationEntry(
        "prosecution", "justice",
        "&{file_charges: +{discovery: &{motions: +{trial: &{closing: +{verdict: end}}}}}}",
        "Prosecution from filing charges through discovery and trial to verdict.",
    ),
    "appeal": CivilizationEntry(
        "appeal", "justice",
        "&{file_appeal: +{brief: &{oral_argument: +{ruling: &{affirm: end, reverse: end}}}}}",
        "Appellate process from filing through oral argument to ruling.",
    ),
    "parole": CivilizationEntry(
        "parole", "justice",
        "rec X . &{hearing: +{grant: &{supervise: +{comply: X, violate: end}}, deny: end}}",
        "Parole as recursive hearing and supervision until violation or completion.",
    ),
    "mediation": CivilizationEntry(
        "mediation", "justice",
        "&{convene: +{opening_statements: &{identify_issues: +{negotiate: &{settle: end, impasse: end}}}}}",
        "Mediation from convening through issue identification to settlement or impasse.",
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def civilization_by_domain(domain: str) -> list[CivilizationEntry]:
    """Return all entries belonging to *domain*.

    Valid domains: governance, economics, education, medicine, technology, justice.
    """
    return [e for e in CIVILIZATION_ENCYCLOPEDIA.values() if e.domain == domain]


def all_civilization_form_lattices() -> bool:
    """Verify that every entry in the civilization encyclopedia forms a lattice.

    Returns True if and only if every entry parses and its state space
    is a lattice.
    """
    for entry in CIVILIZATION_ENCYCLOPEDIA.values():
        ast = parse(entry.session_type_str)
        ss = build_statespace(ast)
        result = check_lattice(ss)
        if not result.is_lattice:
            return False
    return True
