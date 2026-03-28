"""Encyclopedia of Human Experience as Session Types (Step 211).

Every human experience follows a protocol: emotions arise through
recognizable phases, relationships develop through structured interactions,
life events proceed through culturally determined stages, and rituals
encode ancient protocols that connect individuals to community.

The key insight: human experience IS session-typed.  Joy has a protocol
(trigger, rise, peak, afterglow, return).  A friendship has a protocol
(meet, bond, deepen, sustain or drift).  A funeral has a protocol
(gather, mourn, remember, release, depart).  These are not metaphors --
they are the actual structure of lived experience.

This module provides:
    ``HUMAN_ENCYCLOPEDIA``             -- dict of name -> HumanEntry.
    ``human_by_domain(domain)``        -- find entries in a domain.
    ``all_human_form_lattices()``      -- verify every entry forms a lattice.
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
class HumanEntry:
    """A single entry in the human experience encyclopedia.

    Attributes:
        name: Unique identifier for this human experience.
        domain: Experiential domain (emotions, relationships, life_events,
                daily_activities, rituals).
        session_type_str: Session type in textual syntax.
        description: Short human-readable description.
    """

    name: str
    domain: str
    session_type_str: str
    description: str


# ---------------------------------------------------------------------------
# The Human Encyclopedia
# ---------------------------------------------------------------------------

HUMAN_ENCYCLOPEDIA: dict[str, HumanEntry] = {
    # ======================================================================
    # EMOTIONS (12 entries)
    # ======================================================================
    "joy": HumanEntry(
        "joy", "emotions",
        "&{trigger: +{anticipation: &{rise: +{peak: &{savour: +{afterglow: end}}}}}}",
        "Joy unfolds from trigger through anticipation and peak to afterglow.",
    ),
    "sadness": HumanEntry(
        "sadness", "emotions",
        "&{loss_event: +{acknowledge: &{withdraw: +{grieve: &{accept: +{recover: end}}}}}}",
        "Sadness from loss through grief stages to acceptance and recovery.",
    ),
    "anger": HumanEntry(
        "anger", "emotions",
        "&{provocation: +{arousal: &{escalate: +{express: &{cool_down: end}, suppress: &{internalize: end}}}}}",
        "Anger from provocation through arousal to expression or suppression.",
    ),
    "fear": HumanEntry(
        "fear", "emotions",
        "&{threat_detect: +{freeze: &{assess: +{fight: end, flight: end, submit: end}}}}",
        "Fear response: detection, freeze, assessment, and fight/flight/submit.",
    ),
    "surprise": HumanEntry(
        "surprise", "emotions",
        "&{unexpected_event: +{startle: &{reorient: +{evaluate: &{integrate: end}}}}}",
        "Surprise from unexpected event through startle to cognitive integration.",
    ),
    "disgust": HumanEntry(
        "disgust", "emotions",
        "&{noxious_stimulus: +{recoil: &{nausea: +{reject: &{avoid: end}}}}}",
        "Disgust from noxious stimulus through recoil to rejection and avoidance.",
    ),
    "love": HumanEntry(
        "love", "emotions",
        "rec X . &{encounter: +{attraction: &{bond: +{deepen: X, part: end}}}}",
        "Love as recursive deepening of attraction and bonding.",
    ),
    "jealousy": HumanEntry(
        "jealousy", "emotions",
        "&{perceived_threat: +{compare: &{inadequacy: +{confront: end, withdraw: end}}}}",
        "Jealousy from perceived threat through comparison to confrontation or withdrawal.",
    ),
    "pride": HumanEntry(
        "pride", "emotions",
        "&{achievement: +{self_evaluate: &{satisfaction: +{display: &{recognition: end}}}}}",
        "Pride from achievement through self-evaluation to social recognition.",
    ),
    "shame": HumanEntry(
        "shame", "emotions",
        "&{transgression: +{exposed: &{self_condemn: +{hide: &{atone: end}}}}}",
        "Shame from transgression through exposure and self-condemnation to atonement.",
    ),
    "guilt": HumanEntry(
        "guilt", "emotions",
        "&{wrongdoing: +{conscience: &{remorse: +{confess: &{make_amends: +{forgive_self: end}}}}}}",
        "Guilt from wrongdoing through conscience and remorse to amends.",
    ),
    "hope": HumanEntry(
        "hope", "emotions",
        "rec X . &{desire: +{envision: &{strive: +{setback: X, fulfillment: end}}}}",
        "Hope as recursive striving through setbacks toward fulfillment.",
    ),

    # ======================================================================
    # RELATIONSHIPS (8 entries)
    # ======================================================================
    "friendship": HumanEntry(
        "friendship", "relationships",
        "rec X . &{encounter: +{share: &{trust_build: +{deepen: X, drift_apart: end}}}}",
        "Friendship as iterative sharing and trust-building with possible drift.",
    ),
    "romance": HumanEntry(
        "romance", "relationships",
        "&{meet: +{attract: &{court: +{commit: &{grow_together: +{sustain: end, separate: end}}}}}}",
        "Romantic relationship from meeting through courtship to commitment.",
    ),
    "parent_child": HumanEntry(
        "parent_child", "relationships",
        "&{birth: +{nurture: &{teach: +{guide: &{release: +{adult_bond: end}}}}}}",
        "Parent-child arc from birth through nurturing to adult independence.",
    ),
    "sibling": HumanEntry(
        "sibling", "relationships",
        "rec X . &{shared_childhood: +{compete: &{cooperate: +{bond: X, estrange: end}}}}",
        "Sibling relationship cycling between competition and cooperation.",
    ),
    "mentor_mentee": HumanEntry(
        "mentor_mentee", "relationships",
        "&{seek_guidance: +{accept_mentee: &{teach: +{challenge: &{grow: +{surpass: end}}}}}}",
        "Mentoring from guidance-seeking through teaching to student surpassing.",
    ),
    "colleague": HumanEntry(
        "colleague", "relationships",
        "rec X . &{assigned_together: +{collaborate: &{deliver: +{evaluate: X, part_ways: end}}}}",
        "Collegial relationship of repeated collaboration and evaluation.",
    ),
    "rival": HumanEntry(
        "rival", "relationships",
        "rec X . &{challenge: +{compete: &{outcome: +{rematch: X, concede: end, mutual_respect: end}}}}",
        "Rivalry as repeated challenge and competition with resolution.",
    ),
    "stranger_meeting": HumanEntry(
        "stranger_meeting", "relationships",
        "&{notice: +{approach: &{greet: +{converse: &{exchange_info: +{connect: end, part: end}}}}}}",
        "First encounter from noticing through greeting to connection or parting.",
    ),

    # ======================================================================
    # LIFE_EVENTS (10 entries)
    # ======================================================================
    "birth": HumanEntry(
        "birth", "life_events",
        "&{labor_begin: +{contractions: &{deliver: +{first_breath: &{bond_mother: end}}}}}",
        "Birth from labor onset through delivery to first breath and bonding.",
    ),
    "first_words": HumanEntry(
        "first_words", "life_events",
        "&{babble: +{imitate: &{associate: +{utter_word: &{parent_respond: end}}}}}",
        "Language emergence from babbling through imitation to first word.",
    ),
    "first_day_school": HumanEntry(
        "first_day_school", "life_events",
        "&{arrive: +{separate_parent: &{meet_teacher: +{meet_peers: &{first_lesson: +{reunite: end}}}}}}",
        "First school day: separation, new faces, first lesson, reunion.",
    ),
    "graduation": HumanEntry(
        "graduation", "life_events",
        "&{complete_studies: +{ceremony: &{receive_diploma: +{celebrate: &{transition: end}}}}}",
        "Graduation from completion through ceremony to life transition.",
    ),
    "job_interview": HumanEntry(
        "job_interview", "life_events",
        "&{arrive: +{greet_panel: &{present_self: +{answer_questions: &{negotiate: +{accept: end, reject: end}}}}}}",
        "Job interview from arrival through presentation to offer or rejection.",
    ),
    "wedding": HumanEntry(
        "wedding", "life_events",
        "&{prepare: +{gather: &{processional: +{vows: &{rings: +{pronounce: &{celebrate: end}}}}}}}",
        "Wedding ceremony from preparation through vows to celebration.",
    ),
    "parenthood": HumanEntry(
        "parenthood", "life_events",
        "&{conception: +{pregnancy: &{birth_child: +{nurture: &{educate: +{release_child: end}}}}}}",
        "Parenthood arc from conception through nurturing to child independence.",
    ),
    "retirement": HumanEntry(
        "retirement", "life_events",
        "&{last_workday: +{farewell: &{adjust: +{explore: &{settle_routine: end}}}}}",
        "Retirement transition from farewell through adjustment to new routine.",
    ),
    "loss_of_parent": HumanEntry(
        "loss_of_parent", "life_events",
        "&{receive_news: +{shock: &{funeral: +{grieve: &{remember: +{carry_forward: end}}}}}}",
        "Losing a parent: shock, funeral, grieving, and carrying their memory forward.",
    ),
    "death": HumanEntry(
        "death", "life_events",
        "&{decline: +{acceptance: &{farewell: +{release: end}}}}",
        "Death as the final protocol: decline, acceptance, farewell, release.",
    ),

    # ======================================================================
    # DAILY_ACTIVITIES (10 entries)
    # ======================================================================
    "waking_up": HumanEntry(
        "waking_up", "daily_activities",
        "&{alarm: +{stir: &{open_eyes: +{stretch: &{rise: end}}}}}",
        "Morning awakening from alarm through stretching to rising.",
    ),
    "cooking_meal": HumanEntry(
        "cooking_meal", "daily_activities",
        "&{gather_ingredients: +{prep: &{cook: +{plate: &{serve: end}}}}}",
        "Meal preparation from gathering ingredients to plating and serving.",
    ),
    "commuting": HumanEntry(
        "commuting", "daily_activities",
        "&{leave_home: +{travel: &{transfer: +{arrive_work: end}}}}",
        "Daily commute: leaving home, traveling, arriving at work.",
    ),
    "meeting": HumanEntry(
        "meeting", "daily_activities",
        "&{convene: +{set_agenda: &{discuss: +{decide: &{adjourn: end}}}}}",
        "Meeting protocol: convene, set agenda, discuss, decide, adjourn.",
    ),
    "phone_call": HumanEntry(
        "phone_call", "daily_activities",
        "&{ring: +{answer: &{greet: +{converse: &{farewell: +{hang_up: end}}}}}}",
        "Phone call from ring through conversation to farewell.",
    ),
    "shopping": HumanEntry(
        "shopping", "daily_activities",
        "&{enter_store: +{browse: &{select: +{checkout: &{pay: +{leave: end}}}}}}",
        "Shopping from entry through browsing and selection to payment.",
    ),
    "exercising": HumanEntry(
        "exercising", "daily_activities",
        "&{warm_up: +{exert: &{sustain: +{cool_down: &{stretch: end}}}}}",
        "Exercise session: warm up, exertion, sustain, cool down, stretch.",
    ),
    "reading_book": HumanEntry(
        "reading_book", "daily_activities",
        "rec X . &{open_book: +{read_page: &{comprehend: +{turn_page: X, finish: end}}}}",
        "Reading as recursive page-turning until the book is finished.",
    ),
    "falling_asleep": HumanEntry(
        "falling_asleep", "daily_activities",
        "&{lie_down: +{relax: &{drowse: +{drift: &{sleep: end}}}}}",
        "Falling asleep from lying down through drowsiness to sleep.",
    ),
    "dreaming": HumanEntry(
        "dreaming", "daily_activities",
        "rec X . &{rem_phase: +{dream_scene: &{shift: +{continue: X, wake: end}}}}",
        "Dreaming as recursive scene generation during REM sleep.",
    ),

    # ======================================================================
    # RITUALS (10 entries)
    # ======================================================================
    "prayer": HumanEntry(
        "prayer", "rituals",
        "&{kneel: +{invoke: &{petition: +{listen: &{amen: end}}}}}",
        "Prayer protocol: kneel, invoke, petition, listen, close.",
    ),
    "meditation": HumanEntry(
        "meditation", "rituals",
        "rec X . &{sit: +{focus: &{wander: +{notice: X, stillness: end}}}}",
        "Meditation as recursive cycle of focus and wandering toward stillness.",
    ),
    "baptism": HumanEntry(
        "baptism", "rituals",
        "&{gather: +{profess: &{immerse: +{emerge: &{anoint: +{welcome: end}}}}}}",
        "Baptism ritual: gathering, profession, immersion, anointing, welcome.",
    ),
    "wedding_ceremony": HumanEntry(
        "wedding_ceremony", "rituals",
        "&{processional: +{invocation: &{readings: +{vows: &{exchange_rings: +{pronounce: &{recessional: end}}}}}}}",
        "Full wedding ceremony from processional through vows to recessional.",
    ),
    "funeral": HumanEntry(
        "funeral", "rituals",
        "&{gather: +{eulogy: &{remember: +{commit: &{farewell: end}}}}}",
        "Funeral: gathering, eulogizing, remembering, committing, farewell.",
    ),
    "birthday_celebration": HumanEntry(
        "birthday_celebration", "rituals",
        "&{gather_guests: +{surprise: &{cake: +{wish: &{blow_candles: +{gifts: end}}}}}}",
        "Birthday celebration: gathering, cake, wish, candles, gifts.",
    ),
    "new_year": HumanEntry(
        "new_year", "rituals",
        "&{gather: +{countdown: &{midnight: +{celebrate: &{resolve: end}}}}}",
        "New Year celebration: gather, countdown, midnight, celebrate, resolve.",
    ),
    "pilgrimage": HumanEntry(
        "pilgrimage", "rituals",
        "&{prepare: +{depart: &{journey: +{arrive_sacred: &{worship: +{return_home: end}}}}}}",
        "Pilgrimage: preparation, journey, arrival at sacred site, return.",
    ),
    "confession": HumanEntry(
        "confession", "rituals",
        "&{examine_conscience: +{approach_priest: &{confess_sins: +{receive_penance: &{absolution: end}}}}}",
        "Sacramental confession: examination, confession, penance, absolution.",
    ),
    "communion": HumanEntry(
        "communion", "rituals",
        "&{gather: +{liturgy: &{consecrate: +{receive_bread: &{receive_wine: +{blessing: end}}}}}}",
        "Holy Communion: liturgy, consecration, receiving elements, blessing.",
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def human_by_domain(domain: str) -> list[HumanEntry]:
    """Return all entries belonging to *domain*.

    Valid domains: emotions, relationships, life_events, daily_activities, rituals.
    """
    return [e for e in HUMAN_ENCYCLOPEDIA.values() if e.domain == domain]


def all_human_form_lattices() -> bool:
    """Verify that every entry in the human encyclopedia forms a lattice.

    Returns True if and only if every entry parses and its state space
    is a lattice.
    """
    for entry in HUMAN_ENCYCLOPEDIA.values():
        ast = parse(entry.session_type_str)
        ss = build_statespace(ast)
        result = check_lattice(ss)
        if not result.is_lattice:
            return False
    return True
