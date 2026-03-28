"""Encyclopedia of Nature as Session Types (Step 210).

Every natural process follows a protocol: cells divide through regulated
phases, chemical reactions proceed through constrained steps, ecosystems
cycle through predictable stages, and geological forces reshape the Earth
along determined paths.  This module encodes 50+ natural processes as
session types, organized by scientific domain.

The key insight: nature IS protocol.  From DNA replication to hurricane
formation, every natural phenomenon has branch points (environmental
conditions), selection points (system responses), recursion (cycles),
and termination (equilibrium, death, completion).

This module provides:
    ``NATURE_ENCYCLOPEDIA``            -- dict of name -> NatureEntry.
    ``nature_by_domain(domain)``       -- find entries in a domain.
    ``all_nature_form_lattices()``     -- verify every entry forms a lattice.
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
class NatureEntry:
    """A single entry in the nature encyclopedia.

    Attributes:
        name: Unique identifier for this natural process.
        domain: Scientific domain (biology, chemistry, ecology, geology, weather).
        session_type_str: Session type in textual syntax.
        description: Short human-readable description.
    """

    name: str
    domain: str
    session_type_str: str
    description: str


# ---------------------------------------------------------------------------
# The Nature Encyclopedia
# ---------------------------------------------------------------------------

NATURE_ENCYCLOPEDIA: dict[str, NatureEntry] = {
    # ======================================================================
    # BIOLOGY (15 entries)
    # ======================================================================
    "mitosis": NatureEntry(
        "mitosis", "biology",
        "&{interphase: +{prophase: &{metaphase: +{anaphase: &{telophase: +{cytokinesis: end}}}}}}",
        "Cell division producing two identical daughter cells through regulated phases.",
    ),
    "meiosis": NatureEntry(
        "meiosis", "biology",
        "&{interphase: +{meiosis_I: &{crossover: +{separate_homologs: &{meiosis_II: +{separate_chromatids: end}}}}}}",
        "Reductive division producing four haploid gametes with genetic recombination.",
    ),
    "protein_synthesis": NatureEntry(
        "protein_synthesis", "biology",
        "&{transcription: +{splice_mRNA: &{ribosome_bind: +{translate: &{fold: +{release: end}}}}}}",
        "Gene expression from DNA transcription through translation to protein folding.",
    ),
    "dna_replication": NatureEntry(
        "dna_replication", "biology",
        "&{unwind_helicase: +{prime: &{elongate: +{proofread: &{ligate: +{supercoil: end}}}}}}",
        "Semi-conservative DNA duplication with proofreading and ligation.",
    ),
    "apoptosis": NatureEntry(
        "apoptosis", "biology",
        "&{death_signal: +{activate_caspases: &{fragment_dna: +{bleb_membrane: &{phagocytose: end}}}}}",
        "Programmed cell death: orderly self-destruction and cleanup.",
    ),
    "immune_response": NatureEntry(
        "immune_response", "biology",
        "&{detect_pathogen: +{innate_response: &{present_antigen: +{activate_tcells: &{antibody_production: +{clear_pathogen: end, memory_cell: end}}}}}}",
        "Adaptive immune response from pathogen detection to clearance or memory.",
    ),
    "neural_transmission": NatureEntry(
        "neural_transmission", "biology",
        "&{stimulus: +{depolarize: &{action_potential: +{release_neurotransmitter: &{bind_receptor: +{repolarize: end}}}}}}",
        "Neural signal propagation from stimulus to synaptic transmission.",
    ),
    "digestion": NatureEntry(
        "digestion", "biology",
        "&{ingest: +{chew: &{stomach_acid: +{enzyme_break: &{absorb_nutrients: +{excrete: end}}}}}}",
        "Mechanical and chemical breakdown of food into absorbable nutrients.",
    ),
    "respiration": NatureEntry(
        "respiration", "biology",
        "&{inhale_oxygen: +{glycolysis: &{krebs_cycle: +{electron_transport: &{atp_synthesis: +{exhale_co2: end}}}}}}",
        "Cellular respiration converting glucose and oxygen to ATP and CO2.",
    ),
    "photosynthesis": NatureEntry(
        "photosynthesis", "biology",
        "&{absorb_light: +{split_water: &{electron_chain: +{fix_carbon: &{produce_glucose: +{release_oxygen: end}}}}}}",
        "Light-driven conversion of CO2 and water to glucose and oxygen.",
    ),
    "blood_clotting": NatureEntry(
        "blood_clotting", "biology",
        "&{vessel_injury: +{platelet_plug: &{coagulation_cascade: +{fibrin_mesh: &{clot_retraction: +{fibrinolysis: end}}}}}}",
        "Hemostasis: from vessel injury through clot formation to dissolution.",
    ),
    "wound_healing": NatureEntry(
        "wound_healing", "biology",
        "&{injury: +{hemostasis: &{inflammation: +{proliferation: &{remodeling: end}}}}}",
        "Four-phase wound repair from hemostasis through tissue remodeling.",
    ),
    "embryonic_development": NatureEntry(
        "embryonic_development", "biology",
        "&{fertilization: +{cleavage: &{gastrulation: +{organogenesis: &{maturation: +{birth: end}}}}}}",
        "Embryogenesis from fertilization through gastrulation to birth.",
    ),
    "virus_replication": NatureEntry(
        "virus_replication", "biology",
        "&{attach: +{enter_cell: &{uncoat: +{replicate_genome: &{assemble: +{lyse: end, bud: end}}}}}}",
        "Viral lifecycle: attachment, entry, replication, assembly, and release.",
    ),
    "antibiotic_resistance": NatureEntry(
        "antibiotic_resistance", "biology",
        "rec X . &{antibiotic_exposure: +{susceptible_die: end, resistant_survive: &{multiply: +{transfer_gene: X}}}}",
        "Bacterial resistance evolution through selective pressure and gene transfer.",
    ),

    # ======================================================================
    # CHEMISTRY (10 entries)
    # ======================================================================
    "combustion": NatureEntry(
        "combustion", "chemistry",
        "&{fuel_oxygen_mix: +{ignite: &{exothermic_reaction: +{release_heat: &{produce_co2: +{produce_water: end}}}}}}",
        "Rapid oxidation releasing heat, CO2, and water vapor.",
    ),
    "oxidation_reduction": NatureEntry(
        "oxidation_reduction", "chemistry",
        "&{electron_donor: +{oxidize: &{electron_acceptor: +{reduce: &{energy_transfer: end}}}}}",
        "Redox reaction: coupled electron transfer between donor and acceptor.",
    ),
    "acid_base_reaction": NatureEntry(
        "acid_base_reaction", "chemistry",
        "&{acid_proton: +{base_accept: &{form_salt: +{form_water: end}}}}",
        "Neutralization: proton transfer from acid to base producing salt and water.",
    ),
    "precipitation": NatureEntry(
        "precipitation", "chemistry",
        "&{mix_solutions: +{exceed_solubility: &{nucleate: +{crystal_grow: &{settle: end}}}}}",
        "Formation of insoluble solid from supersaturated solution.",
    ),
    "polymerization": NatureEntry(
        "polymerization", "chemistry",
        "rec X . &{monomer_available: +{initiate: &{propagate: +{add_monomer: X, terminate: end}}}}",
        "Chain growth: monomers repeatedly added until termination.",
    ),
    "fermentation": NatureEntry(
        "fermentation", "chemistry",
        "&{glucose: +{glycolysis: &{pyruvate: +{decarboxylate: &{produce_ethanol: +{release_co2: end}}}}}}",
        "Anaerobic glucose metabolism producing ethanol and CO2.",
    ),
    "electrolysis": NatureEntry(
        "electrolysis", "chemistry",
        "&{apply_current: +{cathode_reduce: &{anode_oxidize: +{collect_products: end}}}}",
        "Electrically driven decomposition at cathode and anode.",
    ),
    "catalysis": NatureEntry(
        "catalysis", "chemistry",
        "rec X . &{substrate_bind: +{lower_activation: &{react: +{release_product: &{catalyst_free: X}, deactivate: end}}}}",
        "Catalyst repeatedly lowers activation energy until deactivation.",
    ),
    "crystallization": NatureEntry(
        "crystallization", "chemistry",
        "&{supersaturate: +{nucleate: &{crystal_grow: +{anneal: &{harvest: end}}}}}",
        "Crystal formation from nucleation through growth to harvest.",
    ),
    "nuclear_fission": NatureEntry(
        "nuclear_fission", "chemistry",
        "&{neutron_capture: +{nucleus_split: &{release_neutrons: +{chain_reaction: &{release_energy: end}}}}}",
        "Heavy nucleus splitting releasing neutrons and energy.",
    ),

    # ======================================================================
    # ECOLOGY (10 entries)
    # ======================================================================
    "succession": NatureEntry(
        "succession", "ecology",
        "&{disturbance: +{pioneer_species: &{intermediate_species: +{climax_community: end}}}}",
        "Ecological succession from disturbance through pioneer to climax community.",
    ),
    "migration": NatureEntry(
        "migration", "ecology",
        "rec X . &{season_change: +{depart: &{navigate: +{arrive: &{breed: +{return: X, settle: end}}}}}}",
        "Seasonal animal migration with breeding and return cycle.",
    ),
    "pollination": NatureEntry(
        "pollination", "ecology",
        "&{flower_open: +{attract_pollinator: &{pollinator_visit: +{transfer_pollen: &{fertilize: +{seed_develop: end}}}}}}",
        "Plant reproduction via pollinator-mediated pollen transfer.",
    ),
    "decomposition": NatureEntry(
        "decomposition", "ecology",
        "&{organism_dies: +{bacteria_colonize: &{break_down: +{release_nutrients: &{soil_enrich: end}}}}}",
        "Microbial breakdown of dead matter returning nutrients to soil.",
    ),
    "nitrogen_cycle": NatureEntry(
        "nitrogen_cycle", "ecology",
        "rec X . &{nitrogen_fix: +{nitrify: &{assimilate: +{ammonify: &{denitrify: +{return_atmosphere: X, loss: end}}}}}}",
        "Biogeochemical cycling of nitrogen through fixation and denitrification.",
    ),
    "carbon_cycle": NatureEntry(
        "carbon_cycle", "ecology",
        "rec X . &{photosynthesis_absorb: +{respiration_release: &{decompose_release: +{ocean_absorb: X, sequester: end}}}}",
        "Global carbon cycling between atmosphere, biosphere, and geosphere.",
    ),
    "food_web_energy": NatureEntry(
        "food_web_energy", "ecology",
        "&{sun_energy: +{producer_capture: &{herbivore_consume: +{carnivore_consume: &{decomposer_recycle: end}}}}}",
        "Energy flow through trophic levels from sun to decomposers.",
    ),
    "symbiosis": NatureEntry(
        "symbiosis", "ecology",
        "rec X . &{encounter: +{mutualism: &{both_benefit: X}, separate: end}}",
        "Mutually beneficial interspecies relationship with optional separation.",
    ),
    "parasitism": NatureEntry(
        "parasitism", "ecology",
        "rec X . &{host_encounter: +{attach: &{extract_resource: +{weaken_host: X, host_dies: end}}}}",
        "Parasitic exploitation of host resources until host death.",
    ),
    "competition": NatureEntry(
        "competition", "ecology",
        "&{shared_resource: +{compete: &{dominant_wins: +{subordinate_displaced: end}, coexist: end}}}",
        "Interspecific competition for limited resources with exclusion or coexistence.",
    ),

    # ======================================================================
    # GEOLOGY (8 entries)
    # ======================================================================
    "erosion": NatureEntry(
        "erosion", "geology",
        "rec X . &{weather_expose: +{detach: &{transport: +{deposit: end, continue: X}}}}",
        "Weathering, transport, and deposition of rock material.",
    ),
    "volcanic_eruption": NatureEntry(
        "volcanic_eruption", "geology",
        "&{magma_rise: +{pressure_build: &{vent: +{erupt: &{lava_flow: +{cool: &{solidify: end}}}}}}}",
        "Magma ascent through eruption, lava flow, and solidification.",
    ),
    "earthquake": NatureEntry(
        "earthquake", "geology",
        "&{stress_accumulate: +{fault_rupture: &{seismic_waves: +{ground_shake: &{aftershocks: end}}}}}",
        "Tectonic stress release through fault rupture and seismic wave propagation.",
    ),
    "plate_tectonics": NatureEntry(
        "plate_tectonics", "geology",
        "rec X . &{mantle_convect: +{plate_move: &{converge: +{subduct: X, collide: end}, diverge: +{rift: X, spread: end}}}}",
        "Lithospheric plate motion driven by mantle convection.",
    ),
    "rock_cycle": NatureEntry(
        "rock_cycle", "geology",
        "rec X . &{igneous: +{weather: &{sedimentary: +{metamorphose: &{melt: X}, erode: end}}}}",
        "Transformation between igneous, sedimentary, and metamorphic rock.",
    ),
    "glacier_formation": NatureEntry(
        "glacier_formation", "geology",
        "&{snowfall: +{compact: &{recrystallize: +{flow: &{calve: end}}}}}",
        "Snow compaction into glacial ice, flow, and calving.",
    ),
    "fossilization": NatureEntry(
        "fossilization", "geology",
        "&{organism_dies: +{rapid_burial: &{mineralize: +{lithify: &{expose: end}}}}}",
        "Preservation of remains through burial, mineralization, and lithification.",
    ),
    "weathering": NatureEntry(
        "weathering", "geology",
        "&{rock_expose: +{physical_crack: &{chemical_dissolve: +{biological_break: &{soil_form: end}}}}}",
        "Physical, chemical, and biological breakdown of exposed rock to soil.",
    ),

    # ======================================================================
    # WEATHER (7 entries)
    # ======================================================================
    "hurricane_formation": NatureEntry(
        "hurricane_formation", "weather",
        "&{warm_ocean: +{evaporate: &{convect: +{rotate_coriolis: &{eyewall: +{intensify: &{landfall: +{dissipate: end}}}}}}}}",
        "Tropical cyclone genesis from warm ocean through intensification to landfall.",
    ),
    "tornado": NatureEntry(
        "tornado", "weather",
        "&{supercell: +{wind_shear: &{mesocyclone: +{funnel_descend: &{touchdown: +{dissipate: end}}}}}}",
        "Tornado formation from supercell rotation through funnel descent.",
    ),
    "lightning": NatureEntry(
        "lightning", "weather",
        "&{charge_separate: +{electric_field: &{stepped_leader: +{return_stroke: &{thunder: end}}}}}",
        "Electrical discharge from charge separation through leader and return stroke.",
    ),
    "rainbow_formation": NatureEntry(
        "rainbow_formation", "weather",
        "&{sunlight: +{enter_droplet: &{refract: +{reflect_internal: &{disperse: +{exit_droplet: end}}}}}}",
        "Optical phenomenon from refraction and internal reflection in water droplets.",
    ),
    "snowflake": NatureEntry(
        "snowflake", "weather",
        "&{water_vapor: +{nucleate_ice: &{crystal_grow: +{branch: &{fall: end}}}}}",
        "Ice crystal nucleation and dendritic growth in clouds.",
    ),
    "tide": NatureEntry(
        "tide", "weather",
        "rec X . &{moon_gravity: +{high_tide: &{ebb: +{low_tide: &{flood: X}, neap: end}}}}",
        "Lunar gravitational cycling of ocean tides.",
    ),
    "monsoon": NatureEntry(
        "monsoon", "weather",
        "rec X . &{summer_heat: +{land_low_pressure: &{ocean_moisture: +{heavy_rain: &{flood: +{winter_reverse: X, drought: end}}}}}}",
        "Seasonal wind reversal bringing heavy precipitation.",
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def nature_by_domain(domain: str) -> list[NatureEntry]:
    """Return all entries belonging to *domain*.

    Valid domains: biology, chemistry, ecology, geology, weather.
    """
    return [e for e in NATURE_ENCYCLOPEDIA.values() if e.domain == domain]


def all_nature_form_lattices() -> bool:
    """Verify that every entry in the nature encyclopedia forms a lattice.

    Returns True if and only if every entry parses and its state space
    is a lattice.
    """
    for entry in NATURE_ENCYCLOPEDIA.values():
        ast = parse(entry.session_type_str)
        ss = build_statespace(ast)
        result = check_lattice(ss)
        if not result.is_lattice:
            return False
    return True
