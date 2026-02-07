#!/usr/bin/env python3
"""
AVONET + phylogeny + urban tolerance pipeline

Filtering:
- Species set is defined by presence in the urban intensity dataset
  (via eBird taxonomy crosswalk)

Adds:
- Urban tolerance distance (6 indices)
- Equal weighting across morphology, phylogeny, and urban tolerance
"""

import os
import numpy as np
import pandas as pd
import dendropy


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

BL_PATH = "/home/breallis/datasets/avonet/TraitData/AVONET1_BirdLife.csv"
CROSSWALK_PATH = "/home/breallis/datasets/avonet/PhylogeneticData/BirdLife-BirdTree crosswalk.csv"
PHYLO_PATH = "/home/breallis/datasets/avonet/PhylogeneticData/HackettStage1_0001_1000_MCCTreeTargetHeights.nex"

URBAN_PATH = "/home/breallis/datasets/urban_avian/spp_urban_indices.csv"
EBIRD_CROSSWALK_PATH = "/home/breallis/datasets/ebird_abundances/eBird_taxonomy_v2025.csv"

OUTPUT_FILTERED = "AVONET_Filtered_ByUrbanSpecies.csv"
OUTPUT_COMPARISON = "AVONET_Comparison_WithPhylogeny_Urban.csv"

FOCAL_ID = "AVIBASE-89431E9F"

TRAIT_COLS = [
    "Beak.Length_Culmen",
    "Beak.Length_Nares",
    "Beak.Width",
    "Beak.Depth",
    "Tarsus.Length",
    "Wing.Length",
    "Kipps.Distance",
    "Secondary1",
    "Hand-Wing.Index",
    "Tail.Length",
    "Mass",
]

URBAN_COLS = [
    "Mean.UA",
    "X90th.UA",
    "Block.size.UA",
    "Mean.NL",
    "X90th.NL",
    "Habitat.Use.NL",
]


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def normalize_name(x):
    if pd.isna(x):
        return np.nan
    return x.strip().lower().replace("_", " ")


def standardize(df, cols):
    df = df.copy()
    for c in cols:
        mu = df[c].mean()
        sd = df[c].std()
        df[c] = 0.0 if sd == 0 or np.isnan(sd) else (df[c] - mu) / sd
    return df


def euclidean_distance(df, focal_row, cols, prefix):
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    f = focal_row[cols].apply(pd.to_numeric, errors="coerce")

    valid = X.notna().all(axis=1)
    X = X.loc[valid]
    diffs = X.sub(f.values, axis=1)

    dist = np.sqrt(np.sum(diffs.to_numpy() ** 2, axis=1))

    out = diffs.add_prefix(f"{prefix}_Diff_")
    out[f"{prefix}.Distance"] = dist
    return out.reindex(df.index)


# ------------------------------------------------------------
# Crosswalks
# ------------------------------------------------------------

def load_crosswalk(path):
    cw = pd.read_csv(path)
    cw = cw.dropna(subset=["Species3"])
    return cw.drop_duplicates("Species1", keep="first")


def derive_focal_phylo_label(bl, crosswalk):
    row = bl.loc[bl["Avibase.ID1"] == FOCAL_ID]
    if row.empty:
        raise ValueError("Focal species not found after filtering.")
    sp1 = row.iloc[0]["Species1"]
    sp3 = crosswalk.loc[crosswalk["Species1"] == sp1, "Species3"]
    if sp3.empty:
        raise ValueError("Focal species not found in BirdTree crosswalk.")
    return sp3.iloc[0].replace(" ", "_")


# ------------------------------------------------------------
# Phylogeny
# ------------------------------------------------------------

def compute_phylo_distances(tree, focal_label):
    pdm = tree.phylogenetic_distance_matrix()
    node = tree.find_node_with_taxon_label(focal_label)
    if node is None:
        raise ValueError("Focal species not found in phylogeny.")
    focal = node.taxon
    return {t.label: pdm.distance(focal, t) for t in tree.taxon_namespace}


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    print("Working directory:", os.getcwd())

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    bl = pd.read_csv(BL_PATH, encoding="latin1")
    urban = pd.read_csv(URBAN_PATH)
    ebird = pd.read_csv(EBIRD_CROSSWALK_PATH)

    # --------------------------------------------------------
    # Define species universe via urban dataset
    # --------------------------------------------------------

    urban["species_code"] = urban["species_code"].str.lower()
    ebird["SPECIES_CODE"] = ebird["SPECIES_CODE"].str.lower()

    ebird["sci_norm"] = ebird["SCI_NAME"].apply(normalize_name)
    bl["sci_norm"] = bl["Species1"].apply(normalize_name)

    urban = urban.merge(
        ebird[["SPECIES_CODE", "sci_norm"]],
        left_on="species_code",
        right_on="SPECIES_CODE",
        how="left"
    )

    urban_species = set(urban["sci_norm"].dropna().unique())

    bl = bl.loc[bl["sci_norm"].isin(urban_species)].copy()
    print(f"Filtered AVONET to {len(bl)} species present in urban dataset.")

    if FOCAL_ID not in bl["Avibase.ID1"].values:
        raise RuntimeError("Focal species excluded by urban-species filter.")

    bl.to_csv(OUTPUT_FILTERED, index=False)

    # --------------------------------------------------------
    # Morphology
    # --------------------------------------------------------

    crosswalk = load_crosswalk(CROSSWALK_PATH)
    bl = bl.merge(crosswalk[["Species1", "Species3"]], on="Species1", how="left")
    bl["Species3_underscored"] = bl["Species3"].str.replace(" ", "_")

    bl_morph = standardize(bl, TRAIT_COLS)
    focal_row = bl_morph.loc[bl_morph["Avibase.ID1"] == FOCAL_ID].iloc[0]

    morph_block = euclidean_distance(bl_morph, focal_row, TRAIT_COLS, "Trait")
    bl = pd.concat([bl, morph_block], axis=1)

    # --------------------------------------------------------
    # Urban tolerance
    # --------------------------------------------------------

    bl = bl.merge(
        urban[["sci_norm"] + URBAN_COLS],
        on="sci_norm",
        how="left"
    )

    bl = bl.dropna(subset=URBAN_COLS).copy()
    bl = standardize(bl, URBAN_COLS)

    focal_urban = bl.loc[bl["Avibase.ID1"] == FOCAL_ID].iloc[0]
    urban_block = euclidean_distance(bl, focal_urban, URBAN_COLS, "Urban")
    bl = pd.concat([bl, urban_block], axis=1)

    # --------------------------------------------------------
    # Phylogeny
    # --------------------------------------------------------

    focal_phylo = derive_focal_phylo_label(bl, crosswalk)
    tree = dendropy.Tree.get(
        path=PHYLO_PATH,
        schema="nexus",
        preserve_underscores=True
    )

    phylo_dist = compute_phylo_distances(tree, focal_phylo)
    bl["Phylo.Distance"] = bl["Species3_underscored"].map(phylo_dist)
    bl = bl.dropna(subset=["Phylo.Distance"])

    # --------------------------------------------------------
    # Rank-based combination
    # --------------------------------------------------------

    rank_cols = ["Trait.Distance", "Urban.Distance", "Phylo.Distance"]

    for c in rank_cols:
        # rank 1 = smallest distance = closest
        bl[f"{c}.Rank"] = bl[c].rank(method="average", ascending=True)

    # Mean rank across axes
    bl["Mean.Rank"] = bl[[f"{c}.Rank" for c in rank_cols]].mean(axis=1)

    # Sort by mean rank
    bl = bl.sort_values("Mean.Rank")
    bl.to_csv(OUTPUT_COMPARISON, index=False)

    print(f"Saved rank-based comparison table to {OUTPUT_COMPARISON}")

if __name__ == "__main__":
    main()
