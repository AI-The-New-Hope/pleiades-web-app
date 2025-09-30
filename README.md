[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=20796876&assignment_repo_type=AssignmentRepo)
 # Assignment: Pleiades Star Cluster with Codex CLI

## Introduction (for everyone)

In the night sky, stars are not always spread out evenly — many are born together in groups called **star clusters**.  
- A **star cluster** is a collection of stars that formed from the same giant cloud of gas and dust.  
- Because they are “siblings,” stars in a cluster have almost the same age, distance, and chemical composition.  
- By studying them, astronomers can learn about **how stars are born, live, and die**.  
- Clusters are also important “cosmic laboratories”: they let us test theories of physics and evolution on many stars at once.

The **Pleiades cluster** (also called the *Seven Sisters*) is one of the brightest and closest star clusters. You can see it with the naked eye in winter skies. In this project, you’ll use **Codex CLI** to query real astronomical data from the **Gaia space telescope** and automatically generate Python code to:  
- Find stars around the Pleiades.  
- Use a machine-learning algorithm to separate cluster stars from background stars.  
- Plot what the cluster looks like in the sky, in motion, and in brightness/color.  
- Write up your results on a small GitHub webpage.  

Don’t worry if you’ve never studied astronomy — Codex will help with the coding, and the goal is mostly to **practice GitHub workflow** while getting a taste of real science.

---

**Goal:** Use **OpenAI Codex CLI** to (a) query the **Gaia** archive for stars around the **Pleiades**, (b) find likely cluster members with **DBSCAN/HDBSCAN**, (c) plot results (sky, proper motions, HR diagram), and (d) publish a short **GitHub Pages** report. You should *drive coding via Codex CLI* (you can edit to fix/run).

## Part 1 — Gaia query (Codex)
Generate `scripts/pleiades.py` that:
- Queries Gaia DR3 around **RA≈56.75°, Dec≈24.12°** (cone ~2–3° is fine).
- Retrieves: `ra`, `dec`, `parallax`, `pmra`, `pmdec`, `phot_g_mean_mag`, `bp_rp`.
- Saves a **RA–Dec** scatter plot to `results/pleiades_scatter.png`.

*Tip:* Use `astroquery.gaia` TAP. Gaia stores **parallax** (mas). Distance ≈ 1/(parallax[arcsec]).

## Part 2 — Clustering (Codex)
On a feature branch (e.g., `feature/clustering`), extend the script to:
- Run **DBSCAN** or **HDBSCAN** on kinematics (e.g., `pmra`, `pmdec`, optionally parallax).
- Identify likely cluster members.
- Produce:
  - `results/pleiades_cmd.png` — HR diagram (e.g., G vs BP−RP), highlight members.
  - `results/pleiades_histogram.png` — Parallax histogram.

Merge the branch back to `main` and **tag** your final solution as `v1.0`.

## Part 3 — GitHub Pages report (Codex)
Generate a simple site (e.g., a `docs/index.md`):
- Explain in plain language what the plots show.
- Embed the 3 PNGs from `results/`.
- Enable **GitHub Pages**: Settings → Pages → Build from branch `main`/`docs`.

## Deliverables (must be in your repo)
- `scripts/pleiades.py`
- `results/pleiades_scatter.png`, `results/pleiades_cmd.png`, `results/pleiades_histogram.png`
- A published GitHub Pages site
- Git history showing a feature branch merged, and a tag `v1.0`

## Notes
- Non-astronomy students: focus on workflow. Codex writes code; you keep it running.
- Commit early and often. If stuck, push what you have.
