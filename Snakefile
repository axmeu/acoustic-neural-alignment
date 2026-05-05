"""
Snakefile for the Acoustic research project

Pipeline:
    parse_corpus
        |
    extract_acoustics ──┐
    extract_whisper x layers ──┐
    extract_xlsr    x layers ──┤
                               |
                          normalise (acoustic + each neural layer)
                               |
                          analyse (descriptive, tests, lme, ci_rope, clustering)

Usage:
    snakemake --cores 1                # full pipeline
    snakemake --cores 1 analyse_all    # run only the analyse stages
    snakemake --cores 1 -n             # dry-run, show what would be done
    snakemake --cores 1 --forcerun extract_whisper  # re-run a specific stage
"""
configfile: "config.yaml"

OUT          = config["out_dir"]
NORM_DIR     = config["neural_norm_dir"]
ANALYSIS_DIR = config["analysis_dir"]

WHISPER_LAYERS = config["whisper"]["layers"]
XLSR_LAYERS    = config["xlsr"]["layers"]

# Tags used as sub-directories under outputs/neural_norm/
NEURAL_TAGS = (
    [f"whisper_l{layer}" for layer in WHISPER_LAYERS] +
    [f"xlsr_l{layer}"    for layer in XLSR_LAYERS]
)

# Files written by `normalise_neural` for each tag
NORM_FILES = ["pca_clust.npz", "pca_lme.npz", "pca2.npz", "umap2.npz"]

# Final analysis
ANALYSE_SECTIONS = ["descriptive", "tests", "lme", "ci_rope", "clustering"]


rule all:
    input:
        # one sentinel per analysis section
        expand(f"{ANALYSIS_DIR}/.{{section}}.done",
               section=ANALYSE_SECTIONS),


rule analyse_all:
    """Convenience target to re-run only the analysis sub-sections."""
    input:
        expand(f"{ANALYSIS_DIR}/.{{section}}.done",
               section=ANALYSE_SECTIONS),


# ============================================================================
# 1. Parse corpus
# ============================================================================

rule parse_corpus:
    output:
        f"{OUT}/table.csv"
    params:
        raw_dir       = config["data"]["raw_dir"],
        metadata_file = config["data"]["metadata_file"],
        corr_file     = config["data"]["corr_file"],
        phoneme_tier  = config["data"]["phoneme_tier"],
    shell:
        "pixi run python src/parse_corpus.py "
        "--raw_dir {params.raw_dir} "
        "--metadata_file {params.metadata_file} "
        "--corr_file {params.corr_file} "
        "--phoneme_tier {params.phoneme_tier} "
        "--output {output}"


# ============================================================================
# 2. Acoustic features
# ============================================================================

rule extract_acoustics:
    input:
        table = f"{OUT}/table.csv"
    output:
        f"{OUT}/features_acoustic.csv"
    params:
        n_formants = config["acoustic"]["n_formants"],
    shell:
        "pixi run python src/extract_acoustics.py "
        "--table {input.table} "
        "--output {output} "
        "--n_formants {params.n_formants}"


# ============================================================================
# 3. Whisper extraction per layer
# ============================================================================

rule extract_whisper:
    input: 
        table = f"{OUT}/table.csv"
    output:
        f"{OUT}/features_whisper_l{{layer}}.npz"
    resources:
        gpu=1 
    params:
        model  = config["whisper"]["model"],
        device = config["whisper"]["device"],
    shell:
        "pixi run python src/extract_neural_whisper.py "
        "--table {input.table} "
        "--output {output} "
        "--model {params.model} "
        "--layer {wildcards.layer} "
        "--device {params.device}"


# ============================================================================
# 4. XLSR extraction per layer
# ============================================================================

rule extract_xlsr:
    input:
        table = f"{OUT}/table.csv"
    output:
        f"{OUT}/features_xlsr_l{{layer}}.npz"
    resources:
        gpu=1 
    params:
        model  = config["xlsr"]["model"],
        device = config["xlsr"]["device"],
    shell:
        "pixi run python src/extract_neural_xlsr.py "
        "--table {input.table} "
        "--output {output} "
        "--model {params.model} "
        "--layer {wildcards.layer} "
        "--device {params.device}"


# ============================================================================
# 5. Normalisation
# ============================================================================

rule normalise_acoustic:
    input:
        ac = f"{OUT}/features_acoustic.csv"
    output:
        f"{OUT}/features_acoustic_norm.csv"
    shell:
        "pixi run python src/normalise.py "
        "--acoustic {input.ac} "
        "--output-dir " + OUT

rule normalise_whisper:
    input:
        npz = f"{OUT}/features_whisper_l{{layer}}.npz"
    output:
        expand(f"{NORM_DIR}/whisper_l{{{{layer}}}}/{{fname}}",
               fname=NORM_FILES)
    params:
        n_pca_clust      = config["normalise"]["n_pca_clust"],
        n_pca_lme        = config["normalise"]["n_pca_lme"],
        n_umap_neighbors = config["normalise"]["n_umap_neighbors"],
        umap_min_dist    = config["normalise"]["umap_min_dist"],
    shell:
        "pixi run python src/normalise.py "
        "--whisper {input.npz} "
        "--whisper-tag whisper_l{wildcards.layer} "
        "--output-dir " + NORM_DIR + " "
        "--n-pca-clust {params.n_pca_clust} "
        "--n-pca-lme {params.n_pca_lme} "
        "--n-umap-neighbors {params.n_umap_neighbors} "
        "--umap-min-dist {params.umap_min_dist}"


rule normalise_xlsr:
    input:
        npz = f"{OUT}/features_xlsr_l{{layer}}.npz"
    output:
        expand(f"{NORM_DIR}/xlsr_l{{{{layer}}}}/{{fname}}",
               fname=NORM_FILES)
    params:
        n_pca_clust      = config["normalise"]["n_pca_clust"],
        n_pca_lme        = config["normalise"]["n_pca_lme"],
        n_umap_neighbors = config["normalise"]["n_umap_neighbors"],
        umap_min_dist    = config["normalise"]["umap_min_dist"],
    shell:
        "pixi run python src/normalise.py "
        "--xlsr {input.npz} "
        "--xlsr-tag xlsr_l{wildcards.layer} "
        "--output-dir " + NORM_DIR + " "
        "--n-pca-clust {params.n_pca_clust} "
        "--n-pca-lme {params.n_pca_lme} "
        "--n-umap-neighbors {params.n_umap_neighbors} "
        "--umap-min-dist {params.umap_min_dist}"


# ============================================================================
# 6. Analysis (5 sub-sections)
# ============================================================================

# Each sub-section depends on:
#   - the normalised acoustic CSV
#   - the normalised neural directories (one per tag)
# Snakemake re-runs a sub-section only if one of those changed, OR if the
# corresponding source module is newer than the sentinel.

ANALYSE_INPUTS = (
    [f"{OUT}/features_acoustic_norm.csv"] +
    expand(f"{NORM_DIR}/{{tag}}/{{fname}}",
           tag=NEURAL_TAGS, fname=NORM_FILES)
)


def _analyse_module_path(section):
    """Map a section name to its source module for dependency tracking."""
    return f"src/analyse/{section if section != 'tests' else 'statistical_tests'}.py"


rule analyse_descriptive:
    input:
        ANALYSE_INPUTS,
        module = _analyse_module_path("descriptive"),
    output:
        touch(f"{ANALYSIS_DIR}/.descriptive.done")
    shell:
        "cd src && pixi run python -m analyse.analyse descriptive "
        f"--acoustic ../{OUT}/features_acoustic_norm.csv "
        f"--neural-root ../{NORM_DIR} "
        f"--output ../{ANALYSIS_DIR}"


rule analyse_tests:
    input:
        ANALYSE_INPUTS,
        module = _analyse_module_path("tests"),
    output:
        touch(f"{ANALYSIS_DIR}/.tests.done")
    shell:
        "cd src && pixi run python -m analyse.analyse tests "
        f"--acoustic ../{OUT}/features_acoustic_norm.csv "
        f"--neural-root ../{NORM_DIR} "
        f"--output ../{ANALYSIS_DIR}"


rule analyse_lme:
    input:
        ANALYSE_INPUTS,
        module = _analyse_module_path("lme"),
    output:
        touch(f"{ANALYSIS_DIR}/.lme.done")
    shell:
        "cd src && pixi run python -m analyse.analyse lme "
        f"--acoustic ../{OUT}/features_acoustic_norm.csv "
        f"--neural-root ../{NORM_DIR} "
        f"--output ../{ANALYSIS_DIR}"


rule analyse_ci_rope:
    input:
        ANALYSE_INPUTS,
        module = _analyse_module_path("ci_rope"),
    output:
        touch(f"{ANALYSIS_DIR}/.ci_rope.done")
    shell:
        "cd src && pixi run python -m analyse.analyse ci_rope "
        f"--acoustic ../{OUT}/features_acoustic_norm.csv "
        f"--neural-root ../{NORM_DIR} "
        f"--output ../{ANALYSIS_DIR}"


rule analyse_clustering:
    input:
        ANALYSE_INPUTS,
        module = _analyse_module_path("clustering"),
    output:
        touch(f"{ANALYSIS_DIR}/.clustering.done")
    shell:
        "cd src && pixi run python -m analyse.analyse clustering "
        f"--acoustic ../{OUT}/features_acoustic_norm.csv "
        f"--neural-root ../{NORM_DIR} "
        f"--output ../{ANALYSIS_DIR}"