# ABLIAT - Ablation Laser Impulse Analysis Tool

ABLIAT is a Python-based optimisation framework, built with Streamlit, designed to model and analyse the mission-level feasibility and cost of space-based laser ablation systems for Active Debris Removal (ADR).

This tool finds the most cost-effective system architecture by balancing a complex set of trade-offs. It optimises key parameters (such as aperture diameter, operating range, pulse energy, and repetition rate) to minimise total mission cost, subject to a series of real-world physics and engineering constraints.

## The Problem: Small Debris in LEO

Low Earth Orbit (LEO) is an increasingly congested and contested domain. The growing population of small debris (1-10 cm) poses a disproportionate and mission-ending collision risk to active satellites. While traditional ADR methods (nets, harpoons) are designed for large, single objects, they are not economically viable for removing the millions of smaller fragments.

This tool, ABLIAT, was built to model and analyse a promising solution: **space-based laser ablation**. By imparting a small, contactless impulse from a laser, a debris object's perigee can be lowered until it naturally re-enters and burns up in the atmosphere.

## Core Features

This tool serves as an integrated framework for mission design, allowing an engineer to find the "sweet spot" between cost and performance.

* **Cost-Driven Optimisation:** At its core, the tool uses **SciPy's `differential_evolution`** algorithm to minimise a comprehensive cost model, finding the cheapest system that can perform the mission.
* **Detailed Parametric Cost Model:**
    * **CAPEX:** Models capital costs, including a non-linear aperture cost ($D^{2.3}$) for optics complexity, laser cost ($/kW$) with a fixed floor, and a programmatic "bus" floor.
    * **OPEX:** Models operational costs, including total energy consumption (`$/kWh`) and the "laser-on" time (`$/hour`).
* **Advanced Physics Constraints:** The optimiser is bound by real-world physics:
    * **Thermal Limit:** A hard constraint based on the **Stefan-Boltzmann law**. The optimiser *cannot* select a laser with an average power that produces more waste heat than the user-defined radiator (Area, Temp, Emissivity) can dissipate.
    * **Fluence Band:** A hard constraint that forces the optimiser to find solutions *within* a target fluence band (`F_min`, `F_max`) to ensure efficient ablation without plasma shielding.
* **Debris Population Analysis:** Includes a "roll-up" tool (`integrate_distribution_energy`) to ingest a debris population (e.g., from an ESA MASTER CSV) and calculate the total mission energy, pulses, and cost required to de-orbit the whole catalog.
* **Interactive UI & Visualisation:** Built with Streamlit, the tool provides a full suite of interactive controls and Matplotlib plots to visualise the complex trade-offs between parameters.

## Technical Stack

* **Python 3.10+**
* **Streamlit:** For the interactive web UI.
* **SciPy:** For the `differential_evolution` optimisation engine.
* **Pandas:** For managing debris distribution data.
* **NumPy:** For high-speed numerical calculations.
* **Matplotlib:** For data visualisation.

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Marku1nhos/ABLIAT---Project-Code.git](https://github.com/Marku1nhos/ABLIAT---Project-Code.git)
    cd ABLIAT
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: .\venv\Scripts\activate
    ```

3.  **Install dependencies:**
    The `requirements.txt` file lists all necessary libraries.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the app:**
    ```bash
    streamlit run ABLIAT_v3.0.0.py
    ```

## Sample Data

This repository includes a sample `Debris_dist_data.csv`. This file is a processed 3D matrix of debris distribution by altitude and size, derived from ESA's MASTER model, and can be used with the "Use debris distribution (CSV roll-up)" feature in the app.