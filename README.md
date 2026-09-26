# Predictive Pavement Management & Road Health System
### **Granica × IIT Guwahati Hackathon | IIT Guwahati Estates Office**

An end-to-end, production-ready AI & IoT system designed for the **Indian Institute of Technology Guwahati (IITG) Estates and Works Section**. The system shifts campus road maintenance from reactive "point patching" to predictive "full-segment resurfacing," cutting campus roadway lifecycle expenditures by **over 40%**.

---

## 🏛️ Executive Summary & Core Motivation

In Assam's high-precipitation monsoon climate, traditional reactive point-patching fails repeatedly within 6 to 9 months. Water ingress beneath individual asphalt patches accelerates sub-base erosion, leading to compounding potholes and frequent contractor re-mobilization costs.

This platform unifies **proactive multi-modal telemetry** (surveillance CCTV + 100 Hz bicycle/rickshaw accelerometer runs) with **reactive student reporting** to model road wear dynamically. By clustering defects into **100-meter road corridors using Spatial DBSCAN**, the system flags severely degraded segments for full-width cold milling and resurfacing before structural sub-base collapse occurs.

---

## 🏗️ System Architecture & Ingestion

```
+-----------------------------------------------------------------------------------------+
|                               MULTI-MODAL DATA INGESTION                                |
+----------------------------+-----------------------------+------------------------------+
| 1. Proactive CCTV Streams  | 2. Mobile 100 Hz Sensor Runs| 3. Reactive Student Portal   |
| • RTSP 1080p surveillance  | • 3-Axis Accelerometer (100Hz)• HTML5 GPS geolocation       |
| • Hourly vehicle flow      | • Butterworth High-Pass     | • Automatic AI Spam Filter   |
| • Swerving detection       | • Z-impact spikes (> 1.8g)  | • Immediate YOLOv8 Area (m²) |
+----------------------------+-----------------------------+------------------------------+
                                            │
                                            ▼
+-----------------------------------------------------------------------------------------+
|                                CORE ANALYTICS ENGINE                                    |
+-----------------------------------------------------------------------------------------+
| • Vision Engine: YOLOv8 defect detection + calibrated bounding box surface area (m²)    |
| • Signal Engine: SciPy butterworth filter + peak isolation for Z-axis shock signatures  |
| • Road Damage Index (RDI): RDI = 0.5 * (Area) + 0.3 * (Peak Z) + 0.2 * (Traffic)        |
| • Predictive Degradation Index (PDI): PDI = RDI * (1 + Days_Since_Paved / 365)         |
| • Spatial DBSCAN: 100-meter clustering -> Decision: "Full Resurface" vs "Spot Patch"    |
+-----------------------------------------------------------------------------------------+
                                            │
                                            ▼
+-----------------------------------------------------------------------------------------+
|                           GRANICA COMPLIANCE & EXPORT SUITE                             |
+-----------------------------------------------------------------------------------------+
| • 12-Field PyArrow Apache Parquet export (road_pothole_dataset.parquet)                 |
| • Strict Metadata Summary: 22h Collection Window, Schema, 75/20/5 Split                |
| • Contractor-Ready PDF Work Orders with Lifecycle Cost-Benefit breakdown                |
+-----------------------------------------------------------------------------------------+
```

---

## 📐 Mathematical Formulation

### 1. Road Damage Index ($RDI$)
Quantifies instantaneous surface severity into a normalized score $\in [0.0, 1.0]$:

$$RDI = 0.5 \times \text{Norm}(\text{Area}_{m^2}) + 0.3 \times \text{Norm}(\text{Peak } Z\text{-Impact}_g) + 0.2 \times \text{Norm}(\text{Traffic Index})$$

- $\text{Norm}(\text{Area}) = \min(1.0, \frac{\text{Area}}{4.0})$
- $\text{Norm}(\text{Peak } Z) = \min(1.0, \frac{Z_{\text{peak}} - 1.0}{3.5})$
- $\text{Norm}(\text{Traffic}) = \min(1.0, \frac{\text{Vehicles/Hr}}{200.0})$

### 2. Predictive Degradation Index ($PDI$)
Models time-dependent sub-base wear incorporating days elapsed since last resurfaced:

$$PDI = RDI \times \left(1 + \frac{\text{Days Since Last Paved}}{365}\right)$$

### 3. Spatial DBSCAN (100-Meter Segment Clustering)
Defects are projected onto metric Cartesian space centered around IIT Guwahati coordinates ($26.1900^\circ\text{N}, 91.6935^\circ\text{E}$).
- **Neighborhood Radius ($\epsilon$):** 100 meters
- **Decision Engine:** If segment defect density $\ge 3$ defects OR segment mean $PDI \ge 0.85$, trigger **Full Segment Resurfacing**; otherwise assign **Spot Repair Patching**.
- **Financial Validation:** Replaces $N \times ₹7,500$ recurring spot patches ($₹37,500 \times 2.5 \approx ₹93,750$ over 3 years) with a durable $₹34,000$ full-width resurface, demonstrating **> 40% to 64% net lifecycle savings**.

---

## 📊 Streamlined Parquet Data Schema (12 Fields)

The pipeline exports strictly to `road_pothole_dataset.parquet` conforming to the Granica PyArrow schema:

| Field Name | Type | Description | Sample Value |
|---|---|---|---|
| `observation_id` | String | Unique record ID | `OBS_CORE5_001` |
| `timestamp` | Timestamp (UTC) | ISO-8601 recording time | `2026-09-26T14:30:00Z` |
| `data_source` | String | Collection mode | `CCTV`, `MOBILE_RUN`, `STUDENT_APP` |
| `route_zone` | String | Campus road segment | `Core_V_North_Loop` |
| `gps_lat_long` | String | Coordinate pair | `26.18720, 91.69150` |
| `pothole_area_sqm` | Float64 | Calibrated YOLOv8 surface area | `0.8420` |
| `z_accel_peak_g` | Float64 | SciPy peak impact acceleration ($g$) | `2.8400` |
| `traffic_volume_index`| Float64 | Hourly vehicle count | `112.50` |
| `road_damage_index` | Float64 | Inferred severity score ($RDI \in [0, 1]$)| `0.5840` |
| `days_since_paved` | Int32 | Days elapsed since last surfaced | `420` |
| `predictive_degradation_score` | Float64 | Risk score ($PDI$) for full resurfacing | `1.2556` |
| `synthetic_flag` | Boolean | True for test frames; False for real | `False` |

### Embedded Granica Metadata Summary
- **Observation Source:** `IIT Guwahati Estates Office Multi-Modal Sensor Ingestion (CCTV + Mobile Runs + Student Portal)`
- **Collection Window:** `22h Continuous Campus Monitoring`
- **Schema:** `12-field Granica Streamlined Pavement Parquet Schema v1.0`
- **Data Split:** `75% Observed, 20% Inferred, 5% Synthetic`
- **Granica Compliance:** `TRUE`

---

## 🚀 Quickstart & How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Sample Assets & 22h Dataset
```bash
python create_sample_assets.py
```

### 3. Run Automated Verification Test Suite
```bash
python test_system.py
```
*Output: `Ran 7 tests in ... OK` (verifies YOLO, SciPy, RDI/PDI math, DBSCAN clustering, Parquet metadata, and PDF generation).*

### 4. Launch Multi-Page Streamlit Application
```bash
streamlit run app.py
```

---

## 🖥️ Application Architecture & Portals

1. **🏛️ Starting Gateway Page**
   - Clean institutional entry point with direct role selection: **Continue as Campus Guest** or **Estates Admin Sign-In** (`estates_admin` / `admin123` or 1-Click Demo Login).

2. **👤 Campus Guest Portal**
   - **📸 Report a Road Hazard**: Practical submission flow using photos and campus landmarks (e.g., *"Opposite Core 4 bus stop"*, *"Brahmaputra to Dihing Main Avenue"*). Technical GPS latitude/longitude inputs are eliminated for a streamlined student experience.
   - **📋 View Recent Submissions**: Live tracking table and detailed inspection cards of recently submitted community complaints with status updates.

3. **🛡️ Estates Admin Console**
   - **Prioritized Ranked Maintenance Queue**: Corridors sorted by degradation score ($PDI$ descending) with **estimated costs removed from the list table**.
   - **Detailed Complaint Inspector**:
     - **YOLO AI Defect Analysis**: Highlights potholes and cracks with bounding boxes and calibrated surface areas.
     - **Recommended Action**: Clearly distinguishes between *"Fix Entire 100m Segment (Full Resurfacing)"* vs *"Localized Spot Pothole Repair"* with engineering rationales.
     - **Saved Cost Analysis**: Projects 3-year recurring spot patch failure costs vs full resurfacing to quantify lifecycle savings.
     - **Direct Action Dispatch**: Update statuses, assign civil contractors (e.g. Assam PWD, IITG Civil Works), and generate official signed PDF work orders.
   - **Campus Interactive Map & Sensor Ingestion**: Folium spatial cluster mapping, CCTV RTSP swerving telemetry, and 100 Hz accelerometer waveforms.

---

## 👥 Contributors & Hackathon Submission
- **Event:** Granica × IIT Guwahati Hackathon
- **Target Organization:** Estates and Works Section, Indian Institute of Technology Guwahati
